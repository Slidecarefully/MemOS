# 引入正则模块，用于从字符串中识别特定格式的绑定标记。
# 本文件中主要用于解析 background 里的 [working_binding:<uuid>]。
import re

# 引入 traceback，用于在异常日志中输出完整调用栈，方便排查图数据库或并发任务失败原因。
import traceback

# 引入 uuid，用于在缺少 memory.id 时生成新的图节点 ID。
import uuid

# as_completed 用于等待并发任务完成。
# 它可以按任务完成顺序返回 future，而不是按提交顺序阻塞等待。
from concurrent.futures import as_completed

# datetime 用于记录节点更新时间 updated_at。
# 每次写入图节点时都会把当前时间写入 metadata。
from datetime import datetime

# ContextThreadPoolExecutor 是项目封装过的线程池。
# 相比标准 ThreadPoolExecutor，它通常会额外处理上下文变量传递，保证日志链路或请求上下文不丢失。
from memos.context.context import ContextThreadPoolExecutor

# OllamaEmbedder 是当前 MemoryManager 类型声明中支持的 embedding 组件。
# 它负责把记忆文本转成向量，用于结构节点或检索相关逻辑。
from memos.embedders.factory import OllamaEmbedder

# Neo4jGraphDB 是图数据库封装，MemoryManager 会通过它添加节点、边、查询统计和删除旧节点。
from memos.graph_dbs.neo4j import Neo4jGraphDB

# 支持多种 LLM 实现。
# 这里传给 GraphStructureReorganizer，用于记忆图结构重组时理解和调整节点关系。
from memos.llms.factory import AzureLLM, OllamaLLM, OpenAILLM

# 获取项目统一 logger。
# 这样日志格式、级别和输出目标都与 memos 系统保持一致。
from memos.log import get_logger

# TextualMemoryItem 表示一条文本记忆。
# TreeNodeTextualMemoryMetadata 表示图节点式文本记忆的结构化 metadata。
from memos.memories.textual.item import TextualMemoryItem, TreeNodeTextualMemoryMetadata

# GraphStructureReorganizer 负责在新增节点后异步或延迟重组图结构。
# QueueMessage 是发给 reorganizer 的任务消息，用于说明新增、修改等操作。
from memos.memories.textual.tree_text_memory.organize.reorganizer import (
    GraphStructureReorganizer,
    QueueMessage,
)


# 创建当前模块级 logger。
# __name__ 让日志能标识出来源模块，便于多模块排查。
logger = get_logger(__name__)


# 从增强后的记忆条目中提取 WorkingMemory 绑定 ID。
# 这个函数是“清理临时 WorkingMemory”的辅助入口：先找出绑定关系，再由外部决定是否删除。
def extract_working_binding_ids(mem_items: list[TextualMemoryItem]) -> set[str]:
    """
    Scan enhanced memory items for background hints like
    "[working_binding:<uuid>]" and collect those working memory IDs.

    We store the working<->long binding inside metadata.background when
    initially adding memories in async mode, so we can later clean up
    the temporary WorkingMemory nodes after mem_reader produces the
    final LongTermMemory/UserMemory.

    Args:
        mem_items: list of TextualMemoryItem we just added (enhanced memories)

    Returns:
        A set of working memory IDs (as strings) that should be deleted.
    """
    # 使用 set 保存绑定 ID，可以自然去重。
    # 同一批增强记忆里可能多次引用同一个 WorkingMemory。
    bindings: set[str] = set()

    # 预编译正则表达式，匹配形如 [working_binding:xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx] 的片段。
    # UUID 限定为 36 位常见格式，避免误匹配普通文本。
    pattern = re.compile(r"\[working_binding:([0-9a-fA-F-]{36})\]")

    # 遍历传入的所有记忆 item，从每条记忆的 metadata.background 中尝试提取绑定 ID。
    for item in mem_items:
        try:
            # background 是绑定信息目前存放的位置。
            # getattr 提供默认值，避免 metadata 缺字段时直接失败。
            bg = getattr(item.metadata, "background", "") or ""

        # 如果 metadata 或 background 访问过程异常，这条记忆就视为没有绑定信息。
        except Exception:
            bg = ""

        # background 只有在字符串类型时才适合正则扫描。
        # 如果是列表、字典或其他类型，直接跳过，避免隐式转换造成误匹配。
        if not isinstance(bg, str):
            continue

        # 在 background 中查找第一个 working_binding 标记。
        match = pattern.search(bg)

        # 如果匹配成功，把捕获到的 UUID 加入结果集合。
        if match:
            bindings.add(match.group(1))

    # 返回所有需要后续清理的 WorkingMemory ID。
    return bindings


# MemoryManager 负责把 TextualMemoryItem 写入图数据库，并维护不同记忆类型的容量和结构更新。
# 它处在“记忆抽取结果”和“底层图存储”之间，封装批量写入、并发写入、WorkingMemory 清理和图重组通知。
class MemoryManager:
    # 初始化 MemoryManager 时注入图存储、向量器和 LLM。
    # 这些依赖不会在类内部硬编码创建，方便测试和替换不同后端。
    def __init__(
        self,
        graph_store: Neo4jGraphDB,
        embedder: OllamaEmbedder,
        llm: OpenAILLM | OllamaLLM | AzureLLM,
        memory_size: dict | None = None,
        threshold: float | None = 0.80,
        merged_threshold: float | None = 0.92,
        is_reorganize: bool = False,
    ):
        # 保存图数据库访问对象。
        # 后续节点新增、边操作、统计查询和清理都通过它完成。
        self.graph_store = graph_store

        # 保存 embedding 组件。
        # 它主要用于创建结构节点时生成 key 的向量表示。
        self.embedder = embedder

        # 保存外部传入的容量配置。
        # 如果为空，下面会填充默认容量。
        self.memory_size = memory_size

        # 初始化当前各类记忆的数量缓存。
        # 这不是数据库实时状态，只有调用 _refresh_memory_size 后才会更新。
        self.current_memory_size = {
            "WorkingMemory": 0,
            "LongTermMemory": 0,
            "RawFileMemory": 0,
            "UserMemory": 0,
        }

        # 如果调用方没有提供 memory_size，就使用默认容量上限。
        # 不同记忆类型有不同保留规模：WorkingMemory 最小，长期/文件记忆更大。
        if not memory_size:
            self.memory_size = {
                "WorkingMemory": 20,
                "LongTermMemory": 1500,
                "RawFileMemory": 1500,
                "UserMemory": 480,
            }

        # 输出当前容量配置，便于启动时确认运行参数。
        logger.info(f"MemorySize is {self.memory_size}")

        # 保存普通相似度阈值。
        # 当前文件内没有直接使用，但通常用于判断记忆是否接近、是否需要合并或去重。
        self._threshold = threshold

        # 控制新增节点后是否通知图结构重组器。
        # 关闭时可减少后台整理成本，开启时可以维护更好的图结构。
        self.is_reorganize = is_reorganize

        # 初始化图结构重组器。
        # 它接收 graph_store、LLM、embedder，并根据 is_reorganize 决定是否实际执行重组逻辑。
        self.reorganizer = GraphStructureReorganizer(
            graph_store, llm, embedder, is_reorganize=is_reorganize
        )

        # 保存合并阈值。
        # 当前文件中没有直接使用，但可被后续合并逻辑或重组器相关逻辑引用。
        self._merged_threshold = merged_threshold

    # 对外的统一新增入口。
    # 调用方只需要传入 TextualMemoryItem 列表，不需要关心底层是批量写入还是并行单条写入。
    def add(
        self,
        memories: list[TextualMemoryItem],
        user_name: str | None = None,
        mode: str = "sync",
        use_batch: bool = True,
    ) -> list[str]:
        """
        Add new memories to different memory types.

        Args:
            memories: List of memory items to add.
            user_name: Optional user name for the memories.
            mode: "sync" to cleanup and refresh after adding, "async" to skip.
            use_batch: If True, use batch database operations (more efficient for large batches).
                       If False, use parallel single-node operations (original behavior).

        Returns:
            List of added memory IDs.
        """
        # 初始化返回的节点 ID 列表。
        # 这里返回的是图记忆节点 ID，而不是临时 WorkingMemory 节点 ID。
        added_ids: list[str] = []

        # 默认走批量写入。
        # 批量写入适合较大批次，可以减少数据库往返次数。
        if use_batch:
            added_ids = self._add_memories_batch(memories, user_name)

        # 如果显式关闭批量写入，则使用原始的并发单节点写入逻辑。
        # 这种方式更细粒度，但数据库调用次数更多。
        else:
            added_ids = self._add_memories_parallel(memories, user_name)

        # 同步模式下，新增后立即清理 WorkingMemory，保持短期工作记忆数量不超过上限。
        # 异步模式下跳过清理，通常是为了让后续异步流程还能利用临时节点。
        if mode == "sync":
            self._cleanup_working_memory(user_name)

        # 返回本次真正写入的图记忆节点 ID 列表。
        return added_ids

    # 使用并发方式逐条处理记忆。
    # 这是原始行为：每条记忆单独进入 _process_memory，再由 _process_memory 决定写 WorkingMemory 和图记忆。
    def _add_memories_parallel(
        self, memories: list[TextualMemoryItem], user_name: str | None = None
    ) -> list[str]:
        """
        Add memories using parallel single-node operations (original behavior).
        """
        # 收集所有成功写入的图记忆节点 ID。
        added_ids: list[str] = []

        # 创建最多 10 个工作线程，并发处理多个 memory item。
        with ContextThreadPoolExecutor(max_workers=10) as executor:
            # 为每条记忆提交一个 _process_memory 任务。
            # futures 字典把 future 映射回原始 memory，便于必要时定位失败对象。
            futures = {executor.submit(self._process_memory, m, user_name): m for m in memories}

            # 按完成顺序消费 future。
            # timeout=500 表示整体等待最长 500 秒，防止任务永久卡住。
            for future in as_completed(futures, timeout=500):
                try:
                    # _process_memory 返回该 memory 产生的图记忆 ID 列表。
                    ids = future.result()

                    # 合并到本批次的返回列表。
                    added_ids.extend(ids)

                # 单条记忆处理失败不影响其他记忆写入。
                # 这里记录异常后继续处理剩余 future。
                except Exception as e:
                    logger.exception("Memory processing error: ", exc_info=e)

        # 记录并发写入最终成功的图记忆数量。
        logger.info(f"[MemoryManager: _add_memories_parallel] Added {len(added_ids)} memories")

        # 返回成功写入的图记忆节点 ID。
        return added_ids

    # 使用批量数据库接口写入记忆。
    # 它会先把 TextualMemoryItem 转成图数据库节点字典，再分批提交 add_nodes_batch。
    def _add_memories_batch(
        self, memories: list[TextualMemoryItem], user_name: str | None = None, batch_size: int = 5
    ) -> list[str]:
        """
        Add memories using batch database operations (more efficient for large batches).

        Args:
            memories: List of memory items to add.
            user_name: Optional user name for the memories.
            batch_size: Number of nodes to insert per batch.

        Returns:
            List of added graph memory node IDs.
        """
        # 空列表直接返回，避免创建线程池或调用数据库。
        if not memories:
            return []

        # added_ids 保存对外返回的图记忆节点 ID。
        added_ids: list[str] = []

        # working_nodes 原本用于保存要写入 WorkingMemory 的节点。
        # 但当前代码后面暂时不提交它们，原因见 TODO。
        working_nodes: list[dict] = []

        # graph_nodes 保存真正要写入图记忆空间的节点。
        # 包括 LongTermMemory、UserMemory、RawFileMemory 等。
        graph_nodes: list[dict] = []

        # graph_node_ids 单独保存图节点 ID，用于后续通知 reorganizer。
        graph_node_ids: list[str] = []

        # 遍历每条待写入记忆，把它拆分成可能的 WorkingMemory 节点和图记忆节点。
        for memory in memories:
            # 生成或复用 working_id。
            # 当前写法优先使用 memory.id；如果没有可用 ID，理论上应回退到 uuid。
            # 这里保持原逻辑不改动，只通过注释说明该 ID 会用于 working_binding。
            working_id = memory.id if hasattr(memory, "id") else memory.id or str(uuid.uuid4())

            # 这些类型需要生成一份 WorkingMemory 形态。
            # LongTermMemory/UserMemory 在写入正式图记忆前，也可以先被当作工作记忆进入短期窗口。
            if memory.metadata.memory_type in (
                "WorkingMemory",
                "LongTermMemory",
                "UserMemory",
                "OuterMemory",
            ):
                # 复制原 metadata，并把 memory_type 改成 WorkingMemory。
                # model_copy 保留原字段，update 覆盖记忆类型。
                working_metadata = memory.metadata.model_copy(
                    update={"memory_type": "WorkingMemory"}
                ).model_dump(exclude_none=True)

                # 给节点 metadata 写入更新时间。
                # 图数据库中可通过它判断 FIFO 或最新节点。
                working_metadata["updated_at"] = datetime.now().isoformat()

                # 构造批量写入所需的节点字典。
                # 这里没有直接写库，只是暂存到 working_nodes。
                working_nodes.append(
                    {
                        "id": working_id,
                        "memory": memory.memory,
                        "metadata": working_metadata,
                    }
                )

            # 这些类型需要写入正式图记忆空间。
            # 与 WorkingMemory 不同，它们是检索、组织和长期保留的主要对象。
            if memory.metadata.memory_type in (
                "LongTermMemory",
                "UserMemory",
                "ToolSchemaMemory",
                "ToolTrajectoryMemory",
                "RawFileMemory",
                "SkillMemory",
                "PreferenceMemory",
            ):
                # 生成或复用正式图记忆节点 ID。
                # 这里通常与 memory.id 对齐，从而保持上游创建的 ID 可追踪。
                graph_node_id = (
                    memory.id if hasattr(memory, "id") else memory.id or str(uuid.uuid4())
                )

                # 将 Pydantic metadata 转成普通 dict，方便写入图数据库。
                metadata_dict = memory.metadata.model_dump(exclude_none=True)

                # 更新节点的写入时间。
                metadata_dict["updated_at"] = datetime.now().isoformat()

                # 在正式图记忆中记录它对应的 working_id。
                # 这个绑定关系可用于后续清理临时 WorkingMemory 或追踪生成来源。
                metadata_dict["working_binding"] = working_id

                # Add working_binding for fast mode
                # 读取 tags，用于判断该记忆是否来自 fast 模式。
                tags = metadata_dict.get("tags") or []

                # fast 模式直接从原始输入构造节点，可能没有经过 fine 模式的 LLM 整理。
                # 因此额外打 is_fast 标记，并把 working_binding 写入 background，便于后续识别和清理。
                if "mode:fast" in tags:
                    metadata_dict["is_fast"] = True  # Temporal fix

                    # 保留原 background，不直接覆盖原有摘要或上下文。
                    prev_bg = metadata_dict.get("background", "") or ""

                    # 构造一行可被 extract_working_binding_ids 识别的绑定标记。
                    binding_line = f"[working_binding:{working_id}] direct built from raw inputs"

                    # 如果已有 background，就用分隔符拼接；否则直接使用绑定行。
                    metadata_dict["background"] = (
                        f"{prev_bg} || {binding_line}" if prev_bg else binding_line
                    )

                # 构造正式图记忆节点字典，等待批量提交。
                graph_nodes.append(
                    {
                        "id": graph_node_id,
                        "memory": memory.memory,
                        "metadata": metadata_dict,
                    }
                )

                # 记录正式图节点 ID，供 reorganizer 知道本次新增了哪些节点。
                graph_node_ids.append(graph_node_id)

                # 对外返回的也是正式图节点 ID。
                added_ids.append(graph_node_id)

        # 内部辅助函数：把 nodes 按 batch_size 切分，并发调用 graph_store.add_nodes_batch。
        # node_kind 只用于日志，帮助区分是 WorkingMemory 还是 graph memory 失败。
        def _submit_batches(nodes: list[dict], node_kind: str) -> None:
            # 没有节点时直接返回，避免创建无意义线程池。
            if not nodes:
                return

            # 根据节点数量估算 worker 数。
            # 最多 8 个线程，至少 1 个线程；节点越多，允许更多批次并发提交。
            max_workers = min(8, max(1, len(nodes) // max(1, batch_size)))

            # 使用上下文线程池，保持请求上下文在批量写入线程中可用。
            with ContextThreadPoolExecutor(max_workers=max_workers) as executor:
                # futures 保存 batch 序号、batch 大小和 future，便于失败日志定位。
                futures: list[tuple[int, int, object]] = []

                # 按 batch_size 对节点列表切片。
                for batch_index, i in enumerate(range(0, len(nodes), batch_size), start=1):
                    # 当前批次要写入的节点。
                    batch = nodes[i : i + batch_size]

                    # 提交批量写入任务。
                    fut = executor.submit(
                        self.graph_store.add_nodes_batch, batch, user_name=user_name
                    )

                    # 保存 future 和批次元信息。
                    futures.append((batch_index, len(batch), fut))

                # 等待每个批量写入任务完成。
                for idx, size, fut in futures:
                    try:
                        # 如果底层写入失败，future.result 会抛异常。
                        fut.result()

                    # 单个批次失败时记录日志，但不阻断其他批次结果。
                    except Exception as e:
                        logger.exception(
                            f"Batch add {node_kind} nodes error (batch {idx}, size {size}): ",
                            exc_info=e,
                        )

        # TODO: working id is same with item.id, need to fix, currently stop adding WorkingMemories here.
        #  here used to be: _submit_batches(working_nodes, "WorkingMemory")
        # 当前批量路径下暂时不写入 working_nodes。
        # 因此虽然前面构造了 WorkingMemory 数据，但真正提交的只有 graph_nodes。
        _submit_batches(graph_nodes, "graph memory")

        # 如果新增了正式图节点，并且启用了结构重组，则通知 reorganizer。
        # 这里一次性把本批新增节点 ID 发给重组器，避免每个节点单独触发。
        if graph_node_ids and self.is_reorganize:
            self.reorganizer.add_message(
                QueueMessage(op="add", after_node=graph_node_ids, user_name=user_name)
            )

        # 返回正式图记忆节点 ID 列表。
        return added_ids

    # 清理 WorkingMemory，确保短期工作记忆数量不超过配置上限。
    def _cleanup_working_memory(self, user_name: str | None = None) -> None:
        """
        Remove oldest WorkingMemory nodes to keep within size limit.
        """
        try:
            # 从图数据库删除最旧的 WorkingMemory，只保留最新的指定数量。
            self.graph_store.remove_oldest_memory(
                memory_type="WorkingMemory",
                keep_latest=self.memory_size["WorkingMemory"],
                user_name=user_name,
            )

        # 清理失败不应影响主流程，因此只记录 warning。
        except Exception:
            logger.warning(f"Remove WorkingMemory error: {traceback.format_exc()}")

    # 用给定 memories 替换当前 WorkingMemory 窗口。
    # 这个方法用于刷新短期工作记忆状态，只保留容量上限内的前若干条。
    def replace_working_memory(
        self, memories: list[TextualMemoryItem], user_name: str | None = None
    ) -> None:
        """
        Replace WorkingMemory
        """
        # 只取 WorkingMemory 容量允许的前 N 条作为新的工作记忆候选。
        working_memory_top_k = memories[: self.memory_size["WorkingMemory"]]

        # 并发写入 WorkingMemory 节点。
        with ContextThreadPoolExecutor(max_workers=8) as executor:
            # 每条 memory 都通过 _add_memory_to_db 写成 WorkingMemory 类型。
            futures = [
                executor.submit(
                    self._add_memory_to_db, memory, "WorkingMemory", user_name=user_name
                )
                for memory in working_memory_top_k
            ]

            # 等待所有写入任务完成。
            for future in as_completed(futures, timeout=60):
                try:
                    # 这里只关心任务是否成功，不使用返回 ID。
                    future.result()

                # 单条写入失败时记录异常，但继续等待其他任务。
                except Exception as e:
                    logger.exception("Memory processing error: ", exc_info=e)

        # 写入完成后，再清理超出容量的旧 WorkingMemory。
        self.graph_store.remove_oldest_memory(
            memory_type="WorkingMemory",
            keep_latest=self.memory_size["WorkingMemory"],
            user_name=user_name,
        )

        # 刷新内部数量缓存，让 current_memory_size 与数据库状态尽量一致。
        self._refresh_memory_size(user_name=user_name)

    # 获取当前各类型记忆数量。
    # 该方法不会直接返回旧缓存，而是先从数据库刷新一次。
    def get_current_memory_size(self, user_name: str | None = None) -> dict[str, int]:
        """
        Return the cached memory type counts.
        """
        # 查询图数据库并更新 self.current_memory_size。
        self._refresh_memory_size(user_name=user_name)

        # 返回刷新后的缓存。
        return self.current_memory_size

    # 从图数据库按 memory_type 统计节点数量，并更新本地缓存。
    def _refresh_memory_size(self, user_name: str | None = None) -> None:
        """
        Query the latest counts from the graph store and update internal state.
        """
        # 按 memory_type 分组统计当前用户或命名空间下的节点数量。
        results = self.graph_store.get_grouped_counts(
            group_fields=["memory_type"], user_name=user_name
        )

        # 将数据库返回结果整理成 {memory_type: count} 字典。
        self.current_memory_size = {
            record["memory_type"]: int(record["count"]) for record in results
        }

        # 记录刷新后的容量信息。
        logger.info(f"[MemoryManager] Refreshed memory sizes: {self.current_memory_size}")

    # 处理单条记忆的写入。
    # 它会根据记忆类型决定是否写入 WorkingMemory、是否写入正式图记忆。
    def _process_memory(self, memory: TextualMemoryItem, user_name: str | None = None):
        """
        Process and add memory to different memory types.

        Behavior:
        1. Always create a WorkingMemory node from `memory` and get its node id.
        2. If `memory.metadata.memory_type` is "LongTermMemory" or "UserMemory",
           also create a corresponding long/user node.
           - In async mode, that long/user node's metadata will include
           `working_binding` in `background` which records the WorkingMemory
           node id created in step 1.
        3. Return ONLY the ids of the long/user nodes (NOT the working node id),
           which preserves the previous external contract of `add()`.
        """
        # 返回值只收集正式图记忆节点 ID，不包含 WorkingMemory ID。
        ids: list[str] = []

        # futures 用来保存本条记忆派生出的并发写入任务。
        futures = []

        # TODO: working id is same with item.id, need to fix
        # 为 WorkingMemory 选择 ID。
        # 当前逻辑优先复用 memory.id，这使 WorkingMemory 与正式图节点可能出现相同 ID。
        working_id = memory.id if hasattr(memory, "id") else memory.id or str(uuid.uuid4())

        # 对单条 memory 内部也使用小线程池。
        # 因为它可能同时需要写 WorkingMemory 和正式图记忆，两者可以并行。
        with ContextThreadPoolExecutor(max_workers=2, thread_name_prefix="mem") as ex:
            # 这些类型会被写入 WorkingMemory。
            # 即使原始类型是 LongTermMemory/UserMemory，也会复制成工作记忆形态。
            if memory.metadata.memory_type in (
                "WorkingMemory",
                "LongTermMemory",
                "UserMemory",
                "OuterMemory",
            ):
                # 提交 WorkingMemory 写入任务。
                # forced_id 传入 working_id，保证后续 binding 可以指向这个节点。
                f_working = ex.submit(
                    self._add_memory_to_db, memory, "WorkingMemory", user_name, working_id
                )

                # 标记该 future 是 working 类型，后面不会把它的返回 ID 放进 ids。
                futures.append(("working", f_working))

            # 这些类型会被写入正式图记忆。
            # 包括长期记忆、用户记忆、工具记忆、文件记忆、技能记忆和偏好记忆。
            if memory.metadata.memory_type in (
                "LongTermMemory",
                "UserMemory",
                "ToolSchemaMemory",
                "ToolTrajectoryMemory",
                "RawFileMemory",
                "SkillMemory",
                "PreferenceMemory",
            ):
                # 提交正式图记忆写入任务。
                # working_binding 把该正式节点与前面的 WorkingMemory 节点关联起来。
                f_graph = ex.submit(
                    self._add_to_graph_memory,
                    memory=memory,
                    memory_type=memory.metadata.memory_type,
                    user_name=user_name,
                    working_binding=working_id,
                )

                # 标记为 long，表示它是对外返回的正式图节点。
                futures.append(("long", f_graph))

            # 等待当前 memory 派生出的所有写入任务完成。
            for kind, fut in futures:
                try:
                    # 获取写入结果。
                    res = fut.result()

                    # 只有非 working 的正式图节点 ID 才加入返回列表。
                    if kind != "working" and isinstance(res, str) and res:
                        ids.append(res)

                # 任一写入任务失败时记录完整调用栈。
                # 失败不会再向外抛出，因此本条 memory 可能返回部分成功结果。
                except Exception:
                    logger.warning("Parallel memory processing failed:\n%s", traceback.format_exc())

        # 返回正式图记忆 ID 列表。
        return ids

    # 将一条记忆以指定 memory_type 写入图数据库。
    # 主要用于 WorkingMemory，因为它需要把任意输入 memory 复制成 WorkingMemory 类型。
    def _add_memory_to_db(
        self,
        memory: TextualMemoryItem,
        memory_type: str,
        user_name: str | None = None,
        forced_id: str | None = None,
    ) -> str:
        """
        Add a single memory item to the graph store, with FIFO logic for WorkingMemory.
        If forced_id is provided, use that as the node id.
        """
        # 基于原 metadata 复制一份，并覆盖 memory_type。
        # 这样可以保留 user_id、session_id、tags 等信息，同时切换记忆类型。
        metadata = memory.metadata.model_copy(update={"memory_type": memory_type}).model_dump(
            exclude_none=True
        )

        # 写入更新时间，供后续按时间清理或排序。
        metadata["updated_at"] = datetime.now().isoformat()

        # 如果外部传入 forced_id，就使用它；否则新生成一个 UUID。
        node_id = forced_id or str(uuid.uuid4())

        # 构造要写入的 TextualMemoryItem。
        # 这里 metadata 已经是 dict，保持原代码写法不做改动。
        working_memory = TextualMemoryItem(id=node_id, memory=memory.memory, metadata=metadata)

        # Insert node into graph
        # 调用图数据库接口添加节点。
        # user_name 用于多租户或用户命名空间隔离。
        self.graph_store.add_node(working_memory.id, working_memory.memory, metadata, user_name)

        # 返回写入节点的 ID。
        return node_id

    # 将一条记忆写入正式图记忆空间。
    # 与 _add_memory_to_db 不同，它保留原 memory_type，并会附加 working_binding 与重组器通知。
    def _add_to_graph_memory(
        self,
        memory: TextualMemoryItem,
        memory_type: str,
        user_name: str | None = None,
        working_binding: str | None = None,
    ):
        """
        Generalized method to add memory to a graph-based memory type (e.g., LongTermMemory, UserMemory).
        """
        # 使用 memory.id 作为正式图节点 ID；如果没有 id，则生成 UUID。
        node_id = memory.id if hasattr(memory, "id") else str(uuid.uuid4())

        # Step 2: Add new node to graph
        # 将 metadata 转成 dict，准备写入图数据库。
        metadata_dict = memory.metadata.model_dump(exclude_none=True)

        # 如果传入了 WorkingMemory 绑定 ID，就写进 metadata。
        # 这使正式图节点可以追溯到最初的工作记忆节点。
        if working_binding:
            metadata_dict["working_binding"] = working_binding

        # 根据 tags 判断是否来自 fast 模式。
        tags = metadata_dict.get("tags") or []

        # fast 模式下额外把绑定信息写进 background。
        # 这样 extract_working_binding_ids 能从文本背景中再次解析出关联 WorkingMemory。
        if working_binding and ("mode:fast" in tags):
            metadata_dict["is_fast"] = True  # Temporal fix

            # 读取已有 background，避免覆盖原上下文。
            prev_bg = metadata_dict.get("background", "") or ""

            # 生成标准 working_binding 标记。
            binding_line = f"[working_binding:{working_binding}] direct built from raw inputs"

            # 有旧 background 时拼接；没有则直接赋值。
            if prev_bg:
                metadata_dict["background"] = prev_bg + " || " + binding_line
            else:
                metadata_dict["background"] = binding_line

        # 把正式图记忆节点写入图数据库。
        self.graph_store.add_node(
            node_id,
            memory.memory,
            metadata_dict,
            user_name=user_name,
        )

        # 将新增节点通知给重组器。
        # 注意这里不检查 self.is_reorganize，是否真正处理由 reorganizer 内部或配置决定。
        self.reorganizer.add_message(
            QueueMessage(
                op="add",
                after_node=[node_id],
                user_name=user_name,
            )
        )

        # 返回正式图节点 ID。
        return node_id

    # 将 from_id 上除 MERGED_TO 之外的边迁移到 to_id。
    # 典型场景是记忆合并后，旧节点被归档，但它的上下文关系需要继承给新节点。
    def _inherit_edges(self, from_id: str, to_id: str, user_name: str | None = None) -> None:
        """
        Migrate all non-lineage edges from `from_id` to `to_id`,
        and remove them from `from_id` after copying.
        """
        # 查询 from_id 相关的所有方向、所有类型边。
        edges = self.graph_store.get_edges(
            from_id, type="ANY", direction="ANY", user_name=user_name
        )

        # 逐条迁移边。
        for edge in edges:
            # MERGED_TO 表示合并血缘关系。
            # 这类边需要保留在旧节点上，不能迁移，否则会破坏合并链路。
            if edge["type"] == "MERGED_TO":
                continue  # Keep lineage edges

            # 如果旧边的 from 是 from_id，则迁移后 from 改成 to_id。
            # 否则 from 保持原样。
            new_from = to_id if edge["from"] == from_id else edge["from"]

            # 如果旧边的 to 是 from_id，则迁移后 to 改成 to_id。
            # 否则 to 保持原样。
            new_to = to_id if edge["to"] == from_id else edge["to"]

            # 如果迁移后形成自环，就跳过。
            # 自环通常没有业务意义，也可能影响图算法。
            if new_from == new_to:
                continue

            # Add edge to merged node if it doesn't already exist
            # 如果新边不存在，则创建迁移后的边。
            # 先查存在性可以避免重复边。
            if not self.graph_store.edge_exists(
                new_from, new_to, edge["type"], direction="ANY", user_name=user_name
            ):
                self.graph_store.add_edge(new_from, new_to, edge["type"], user_name=user_name)

            # Remove original edge if it involved the archived node
            # 删除旧边，完成从旧节点到新节点的关系迁移。
            self.graph_store.delete_edge(
                edge["from"], edge["to"], edge["type"], user_name=user_name
            )

    # 确保某个结构路径或结构节点存在。
    # 当前实现根据 metadata.key 查找或创建一个结构节点，并返回该节点 ID。
    def _ensure_structure_path(
        self,
        memory_type: str,
        metadata: TreeNodeTextualMemoryMetadata,
        user_name: str | None = None,
    ) -> str:
        """
        Ensure structural path exists (ROOT → ... → final node), return last node ID.

        Args:
            memory_type: Memory type for the structure node.
            metadata: Metadata containing key and other fields.
            user_name: Optional user name for multi-tenant isolation.

        Returns:
            Final node ID of the structure path.
        """
        # Step 1: Try to find an existing memory node with content == tag
        # 先按 memory 和 memory_type 查询是否已有结构节点。
        # 这里 memory 字段使用 metadata.key，表示结构节点以 key 作为内容。
        existing = self.graph_store.get_by_metadata(
            [
                {"field": "memory", "op": "=", "value": metadata.key},
                {"field": "memory_type", "op": "=", "value": memory_type},
            ],
            user_name=user_name,
        )

        # 如果已存在匹配节点，直接复用第一个节点。
        if existing:
            node_id = existing[0]  # Use the first match

        # 如果不存在，就创建一个新的结构节点。
        else:
            # Step 2: If not found, create a new structure node
            # 结构节点本身也是 TextualMemoryItem，但 memory 内容就是 metadata.key。
            new_node = TextualMemoryItem(
                memory=metadata.key,
                metadata=TreeNodeTextualMemoryMetadata(
                    # 继承原 metadata 中的用户和会话信息。
                    user_id=metadata.user_id,
                    session_id=metadata.session_id,

                    # 使用调用方传入的 memory_type 作为结构节点类型。
                    memory_type=memory_type,

                    # 新结构节点默认激活。
                    status="activated",

                    # 结构节点不继承 tags，避免混入普通记忆标签。
                    tags=[],

                    # key 同样设置为 metadata.key。
                    key=metadata.key,

                    # 为结构节点内容生成 embedding，方便后续结构检索或相似度计算。
                    embedding=self.embedder.embed([metadata.key])[0],

                    # 初始化使用记录为空。
                    usage=[],

                    # 结构节点不是直接来自原始消息，因此 sources 为空。
                    sources=[],

                    # 默认置信度。
                    confidence=0.99,

                    # 结构节点背景为空。
                    background="",
                ),
            )

            # 将新结构节点写入图数据库。
            self.graph_store.add_node(
                new_node.id,
                new_node.memory,
                new_node.metadata.model_dump(exclude_none=True),
                user_name=user_name,
            )

            # 通知重组器有新结构节点加入。
            self.reorganizer.add_message(
                QueueMessage(
                    op="add",
                    after_node=[new_node.id],
                    user_name=user_name,
                )
            )

            # 保存新节点 ID，作为返回值。
            node_id = new_node.id

        # Step 3: Return this structure node ID as the parent_id
        # 返回结构路径最后一个节点 ID。
        # 当前实现中它就是找到或创建的结构节点 ID。
        return node_id

    # 对外的清理并刷新入口。
    # 它先按容量策略清理旧记忆，再刷新当前数量缓存。
    def remove_and_refresh_memory(self, user_name: str | None = None):
        # 根据当前容量和阈值决定是否执行清理。
        self._cleanup_memories_if_needed(user_name=user_name)

        # 清理后重新统计各类记忆数量。
        self._refresh_memory_size(user_name=user_name)

    # 如果某类记忆数量接近容量上限，就清理最旧节点。
    # 这是一种“接近满才清理”的策略，避免每次新增都触发数据库删除。
    def _cleanup_memories_if_needed(self, user_name: str | None = None) -> None:
        """
        Only clean up memories if we're close to or over the limit.
        This reduces unnecessary database operations.
        """
        # 达到容量 80% 时开始清理。
        cleanup_threshold = 0.8  # Clean up when 80% full

        # 输出容量配置，便于理解后续清理判断。
        logger.info(f"self.memory_size: {self.memory_size}")

        # 遍历每一种记忆类型及其容量上限。
        for memory_type, limit in self.memory_size.items():
            # 从缓存中读取当前数量；没有记录时默认为 0。
            current_count = self.current_memory_size.get(memory_type, 0)

            # 计算触发清理的数量阈值。
            threshold = int(int(limit) * cleanup_threshold)

            # Only clean up if we're at or above the threshold
            # 只有当前数量达到阈值时才清理。
            if current_count >= threshold:
                try:
                    # 删除最旧节点，只保留 limit 条最新记忆。
                    self.graph_store.remove_oldest_memory(
                        memory_type=memory_type, keep_latest=limit, user_name=user_name
                    )

                    # debug 级别记录清理动作。
                    logger.debug(f"Cleaned up {memory_type}: {current_count} -> {limit}")

                # 单类记忆清理失败不影响其他类型继续尝试。
                except Exception:
                    logger.warning(f"Remove {memory_type} error: {traceback.format_exc()}")

    # 等待图结构重组器完成当前已提交的任务。
    # 通常在关闭资源或测试中使用，确保后台整理任务已经落定。
    def wait_reorganizer(self):
        """
        Wait for the reorganizer to finish processing all messages.
        """
        # 记录等待动作。
        logger.debug("Waiting for reorganizer to finish processing messages...")

        # 阻塞直到当前任务队列处理完毕。
        self.reorganizer.wait_until_current_task_done()

    # 关闭 MemoryManager 持有的后台资源。
    # 当前主要是等待并停止 reorganizer。
    def close(self):
        # 先等待已提交重组任务完成，避免直接停止造成任务丢失。
        self.wait_reorganizer()

        # 停止重组器后台工作线程或资源。
        self.reorganizer.stop()

    # 析构函数在对象被垃圾回收时尝试关闭资源。
    # 它是最后兜底，不应替代显式 close。
    def __del__(self):
        # 调用 close，尽量释放 reorganizer 资源。
        self.close()
