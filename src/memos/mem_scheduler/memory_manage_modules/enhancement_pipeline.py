# 启用 postponed evaluation of annotations。
# 这样类型注解不会在函数定义阶段立即求值，可减少循环引用和运行时导入压力。
from __future__ import annotations

# time 用于增强失败后的简单重试等待。
import time

# TYPE_CHECKING 用于只在类型检查阶段导入某些类型，避免运行时额外依赖。
from typing import TYPE_CHECKING

# 获取模块级 logger，用于记录增强、重试、解析失败等运行信息。
from memos.log import get_logger

# 调度检索增强的默认批大小。
# 当配置对象没有 scheduler_retriever_batch_size 时使用它。
from memos.mem_scheduler.schemas.general_schemas import (
    DEFAULT_SCHEDULER_RETRIEVER_BATCH_SIZE,

    # 调度检索增强的默认重试次数。
    # 当配置对象没有 scheduler_retriever_enhance_retries 时使用它。
    DEFAULT_SCHEDULER_RETRIEVER_RETRIES,
)

# extract_json_obj 用于从 LLM 文本响应中提取 JSON 对象。
# extract_list_items_in_answer 用于从 LLM 响应中提取列表项形式的增强 memory 文本。
from memos.mem_scheduler.utils.misc_utils import extract_json_obj, extract_list_items_in_answer

# TextualMemoryItem 是文本记忆条目的统一数据结构。
# TextualMemoryMetadata 是新建增强记忆时使用的元数据结构。
from memos.memories.textual.item import TextualMemoryItem, TextualMemoryMetadata

# FINE_STRATEGY 是当前 fine 阶段的全局策略配置。
# FineStrategy 是枚举类型，用来区分 RECREATE 和 REWRITE 等策略。
from memos.types.general_types import FINE_STRATEGY, FineStrategy


# 当前模块 logger。
logger = get_logger(__name__)

# TYPE_CHECKING 为 True 时才导入 Callable。
# 这样运行时不会因为类型注解而增加导入成本。
if TYPE_CHECKING:
    # Callable 表示可调用对象。
    # EnhancementPipeline 接收 build_prompt 作为一个 prompt 构造函数。
    from collections.abc import Callable


# EnhancementPipeline 负责“基于 query 对记忆进行二次增强”的流水线。
# 它不直接负责调度，也不直接写数据库；它更像一个纯处理模块：
# 输入 query history + memory items，调用 LLM 生成增强后的 memory items。
class EnhancementPipeline:
    # 初始化增强流水线。
    # process_llm 负责实际生成；config 提供批大小和重试次数；build_prompt 负责根据模板构造 prompt。
    def __init__(self, process_llm, config, build_prompt: Callable[..., str]):
        # 保存用于增强、评估和召回提示生成的 LLM。
        self.process_llm = process_llm

        # 保存配置对象。
        # 后续从中读取 scheduler_retriever_batch_size 和 scheduler_retriever_enhance_retries。
        self.config = config

        # 保存 prompt 构造函数。
        # 该函数通过 template_name/prompt_name 和变量组装最终 prompt。
        self.build_prompt = build_prompt

        # 从配置读取增强批大小。
        # 如果配置缺失该字段，则使用默认 DEFAULT_SCHEDULER_RETRIEVER_BATCH_SIZE。
        self.batch_size: int | None = getattr(
            config, "scheduler_retriever_batch_size", DEFAULT_SCHEDULER_RETRIEVER_BATCH_SIZE
        )

        # 从配置读取增强失败后的重试次数。
        # 如果配置缺失该字段，则使用默认 DEFAULT_SCHEDULER_RETRIEVER_RETRIES。
        self.retries: int = getattr(
            config, "scheduler_retriever_enhance_retries", DEFAULT_SCHEDULER_RETRIEVER_RETRIES
        )

    # 判断当前已有 memory 是否足以回答某个 query。
    # 返回 True 表示现有记忆足够，False 表示可能需要扩大召回或继续处理。
    def evaluate_memory_answer_ability(
        self, query: str, memory_texts: list[str], top_k: int | None = None
    ) -> bool:
        # 如果指定 top_k，只评估前 top_k 条 memory；否则评估全部 memory_texts。
        # 这能避免把过多记忆放进 prompt，降低 LLM 判断成本。
        limited_memories = memory_texts[:top_k] if top_k is not None else memory_texts

        # 构造“记忆是否足以回答问题”的评估 prompt。
        # 如果没有可用记忆，则明确传入 "No memories available"。
        prompt = self.build_prompt(
            template_name="memory_answer_ability_evaluation",
            query=query,
            memory_list="\n".join([f"- {memory}" for memory in limited_memories])
            if limited_memories
            else "No memories available",
        )

        # 调用 LLM 进行评估。
        # 这里期望 LLM 返回可解析的 JSON，且包含 result 字段。
        response = self.process_llm.generate([{"role": "user", "content": prompt}])

        try:
            # 从 LLM 响应中提取 JSON 对象。
            result = extract_json_obj(response)

            # 如果 JSON 中有 result 字段，则把它作为最终布尔判断返回。
            if "result" in result:
                logger.info(
                    "Answerability: result=%s; reason=%s; evaluated=%s",
                    result["result"],
                    result.get("reason", "n/a"),
                    len(limited_memories),
                )
                return result["result"]

            # 如果 JSON 结构不符合预期，则记录 warning 并返回 False。
            # False 会让上游倾向于继续召回或增强，属于保守策略。
            logger.warning("Answerability: invalid LLM JSON structure; payload=%s", result)
            return False

        # JSON 解析或结构读取失败时返回 False。
        # 这样不会因为 LLM 输出格式波动而中断主流程。
        except Exception as e:
            logger.error("Answerability: parse failed; err=%s; raw=%s...", e, str(response)[:200])
            return False

    # 构造 memory enhancement prompt。
    # 它会根据 query_history 数量和 FINE_STRATEGY 选择不同模板与 memory 展示格式。
    def _build_enhancement_prompt(self, query_history: list[str], batch_texts: list[str]) -> str:
        # 如果只有一个 query，就直接把列表中的字符串取出来。
        # 这样模板中 query_history 位置会是一个普通字符串，而不是列表。
        if len(query_history) == 1:
            query_history = query_history[0]

        # 如果有多个 query，则把每个 query 编号，保留历史顺序。
        else:
            query_history = (
                [f"[{i}] {query}" for i, query in enumerate(query_history)]
                if len(query_history) > 1
                else query_history[0]
            )

        # REWRITE 策略表示“基于原 memory 改写”。
        # 因为后续需要把 LLM 输出映射回原 memory，所以这里给每条 memory 加索引。
        if FINE_STRATEGY == FineStrategy.REWRITE:
            text_memories = "\n".join([f"- [{i}] {mem}" for i, mem in enumerate(batch_texts)])
            prompt_name = "memory_rewrite_enhancement"

        # 非 REWRITE 策略走 RECREATE 模板。
        # 这里不要求输出与原索引一一对应，更像是让 LLM 根据输入重建长期记忆。
        else:
            text_memories = "\n".join([f"- {mem}" for i, mem in enumerate(batch_texts)])
            prompt_name = "memory_recreate_enhancement"

        # 调用统一 prompt builder 生成最终 prompt。
        # prompt 内会包含 query_history 和本批待增强 memories。
        return self.build_prompt(
            prompt_name,
            query_history=query_history,
            memories=text_memories,
        )

    # 处理一个批次的 memory enhancement。
    # 返回增强后的 memory 列表，以及当前批次是否成功。
    def _process_enhancement_batch(
        self,
        batch_index: int,
        query_history: list[str],
        memories: list[TextualMemoryItem],
        retries: int,
    ) -> tuple[list[TextualMemoryItem], bool]:
        # attempt 记录当前尝试次数。
        attempt = 0

        # 只取每个 TextualMemoryItem 的 memory 文本，作为 LLM 输入。
        text_memories = [one.memory for one in memories]

        # 为当前批次构造增强 prompt。
        prompt = self._build_enhancement_prompt(
            query_history=query_history, batch_texts=text_memories
        )

        # 保存最近一次 LLM 响应。
        # 如果所有重试失败，日志会输出该响应辅助排查。
        llm_response = None

        # 最多尝试 max(0, retries) + 2 次。
        # 注意这里是 <= max(0, retries) + 1，因此实际次数比 retries 字面值多一次兜底尝试。
        while attempt <= max(0, retries) + 1:
            try:
                # 调用 LLM 生成增强后的记忆文本。
                llm_response = self.process_llm.generate([{"role": "user", "content": prompt}])

                # 从 LLM 响应中抽取列表项。
                # 这里假设增强结果以列表形式返回，例如多行 bullet 或编号项。
                processed_text_memories = extract_list_items_in_answer(llm_response)

                # 如果抽取到了至少一条增强记忆，则认为本次 LLM 生成有效。
                if len(processed_text_memories) > 0:
                    # enhanced_memories 保存转换回 TextualMemoryItem 的结果。
                    enhanced_memories = []

                    # 从原始 memory metadata 中继承 user_id。
                    # 当前批次至少有一条 memory，否则上游不会调用到这里。
                    user_id = memories[0].metadata.user_id

                    # RECREATE 策略：把 LLM 输出视为新长期记忆。
                    # 新对象没有继承原 id，因为它们是“重新创建”的 memory。
                    if FINE_STRATEGY == FineStrategy.RECREATE:
                        # 遍历 LLM 生成的新 memory 文本。
                        for new_mem in processed_text_memories:
                            # 创建新的 TextualMemoryItem。
                            # memory_type 固定设为 LongTermMemory，表示增强后进入长期记忆层。
                            enhanced_memories.append(
                                TextualMemoryItem(
                                    memory=new_mem,
                                    metadata=TextualMemoryMetadata(
                                        user_id=user_id, memory_type="LongTermMemory"
                                    ),
                                )
                            )

                    # REWRITE 策略：把 LLM 输出视为对原 memory 的改写。
                    # 这种情况下应尽量保留原 id 和原 metadata，只替换 memory 文本。
                    elif FINE_STRATEGY == FineStrategy.REWRITE:

                        # 解析 LLM 输出项中的索引和文本。
                        # 支持 "[0] xxx"、"0: xxx"、"0- xxx"、"0) xxx" 等格式。
                        def _parse_index_and_text(s: str) -> tuple[int | None, str]:
                            # 局部导入 re，只有 REWRITE 路径需要正则。
                            import re

                            # 先规整空值和首尾空格。
                            s = (s or "").strip()

                            # 匹配 "[数字] 文本"。
                            m = re.match(r"^\s*\[(\d+)\]\s*(.+)$", s)
                            if m:
                                return int(m.group(1)), m.group(2).strip()

                            # 匹配 "数字: 文本"、"数字- 文本"、"数字) 文本"。
                            m = re.match(r"^\s*(\d+)\s*[:\-\)]\s*(.+)$", s)
                            if m:
                                return int(m.group(1)), m.group(2).strip()

                            # 如果没有索引，就返回 None 和原文本。
                            return None, s

                        # 建立 index -> 原 memory 的映射。
                        # 这与 _build_enhancement_prompt 中 "[i]" 索引格式配套。
                        idx_to_original = dict(enumerate(memories))

                        # 遍历 LLM 输出项。
                        for j, item in enumerate(processed_text_memories):
                            # 尝试从输出文本中解析原始索引和改写文本。
                            idx, new_text = _parse_index_and_text(item)

                            # 如果解析到了有效索引，就用该索引找到原 memory。
                            if idx is not None and idx in idx_to_original:
                                orig = idx_to_original[idx]

                            # 如果 LLM 没有输出索引，则按输出顺序与原 memories 对齐。
                            else:
                                orig = memories[j] if j < len(memories) else None

                            # 如果找不到对应原 memory，则跳过这一条输出。
                            if not orig:
                                continue

                            # 创建改写后的 TextualMemoryItem。
                            # 保留原 id 和 metadata，只替换 memory 文本。
                            enhanced_memories.append(
                                TextualMemoryItem(
                                    id=orig.id,
                                    memory=new_text,
                                    metadata=orig.metadata,
                                )
                            )

                    # FINE_STRATEGY 不是 RECREATE/REWRITE 时，记录错误。
                    else:
                        logger.error("Fine search strategy %s not exists", FINE_STRATEGY)

                    # 成功生成增强结果后记录完整日志。
                    # 注意日志里包含 prompt 和 llm_response，生产环境可能需要关注敏感信息。
                    logger.info(
                        "[enhance_memories_with_query] done | Strategy=%s | prompt=%s | llm_response=%s",
                        FINE_STRATEGY,
                        prompt,
                        llm_response,
                    )

                    # 当前批次成功，返回增强后的 memory 列表。
                    return enhanced_memories, True

                # 如果列表解析结果为空，主动抛异常进入重试。
                raise ValueError(
                    "Fail to run memory enhancement; retry "
                    f"{attempt}/{max(1, retries) + 1}; "
                    f"processed_text_memories: {processed_text_memories}"
                )

            # 捕获当前尝试中的任何异常，准备下一次重试。
            except Exception as e:
                # 尝试次数加一。
                attempt += 1

                # 简单等待 1 秒后重试。
                time.sleep(1)

                # 记录 debug 级别重试日志。
                logger.debug(
                    "[enhance_memories_with_query][batch=%s] retry %s/%s failed: %s",
                    batch_index,
                    attempt,
                    max(1, retries) + 1,
                    e,
                )

        # 所有尝试都失败后记录 error。
        logger.error(
            "Fail to run memory enhancement; prompt: %s;\n llm_response: %s",
            prompt,
            llm_response,
            exc_info=True,
        )

        # 失败时返回原 memories，并标记 False。
        # 这样上游可以选择继续使用原记忆，而不是直接丢失数据。
        return memories, False

    # 将 memory 列表按 batch_size 切分成多个批次。
    # 每个批次包含起始下标、结束下标和对应 memory 子列表。
    @staticmethod
    def _split_batches(
        memories: list[TextualMemoryItem], batch_size: int
    ) -> list[tuple[int, int, list[TextualMemoryItem]]]:
        # batches 收集所有切分结果。
        batches: list[tuple[int, int, list[TextualMemoryItem]]] = []

        # start 表示当前批次起点。
        start = 0

        # n 是总 memory 数。
        n = len(memories)

        # 只要 start 没到末尾，就继续切分。
        while start < n:
            # end 不超过列表长度。
            end = min(start + batch_size, n)

            # 添加当前批次。
            batches.append((start, end, memories[start:end]))

            # 下一批从当前 end 开始。
            start = end

        # 返回所有批次。
        return batches

    # 当现有 memory 不足以回答 query 时，让 LLM 给出扩大召回的 hint。
    # 返回 hint 文本和是否触发召回的布尔值。
    def recall_for_missing_memories(self, query: str, memories: list[str]) -> tuple[str, bool]:
        # 把 memory 列表格式化为 prompt 中的 bullet list。
        text_memories = "\n".join([f"- {mem}" for i, mem in enumerate(memories)])

        # 构造扩大召回 prompt。
        # LLM 应返回 JSON，包含 hint 和 trigger_recall 等字段。
        prompt = self.build_prompt(
            template_name="enlarge_recall",
            query=query,
            memories_inline=text_memories,
        )

        # 调用 LLM 生成召回 hint。
        llm_response = self.process_llm.generate([{"role": "user", "content": prompt}])

        # 从 LLM 响应中解析 JSON。
        json_result: dict = extract_json_obj(llm_response)

        # 记录 prompt 和响应，便于排查召回提示质量。
        logger.info(
            "[recall_for_missing_memories] done | prompt=%s | llm_response=%s",
            prompt,
            llm_response,
        )

        # 读取 hint 字段，缺省为空字符串。
        hint = json_result.get("hint", "")

        # 没有 hint 时，不触发扩大召回。
        if len(hint) == 0:
            return hint, False

        # 有 hint 时，同时返回 trigger_recall 字段。
        # 如果 JSON 中没有该字段，默认 False。
        return hint, json_result.get("trigger_recall", False)

    # 对一组 memories 按 query_history 做增强。
    # 它会根据 batch_size 决定单批处理还是多批并发处理，并在失败时尽量回退到原始 memories。
    def enhance_memories_with_query(
        self,
        query_history: list[str],
        memories: list[TextualMemoryItem],
    ) -> tuple[list[TextualMemoryItem], bool]:
        # 如果没有 memories，直接跳过增强。
        # 返回 True 表示流程本身没有失败，只是没有数据可处理。
        if not memories:
            logger.warning("[Enhance] skipped (no memories to process)")
            return memories, True

        # 读取实例初始化时确定的批大小。
        batch_size = self.batch_size

        # 读取实例初始化时确定的重试次数。
        retries = self.retries

        # 记录待增强 memory 数量。
        num_of_memories = len(memories)

        try:
            # 如果没有设置 batch_size，或者 memory 数不超过 batch_size，则单批处理。
            if batch_size is None or num_of_memories <= batch_size:
                # 直接调用单批增强函数。
                enhanced_memories, success_flag = self._process_enhancement_batch(
                    batch_index=0,
                    query_history=query_history,
                    memories=memories,
                    retries=retries,
                )

                # 单批场景下，总成功标记就是该批的成功标记。
                all_success = success_flag

            # 如果 memory 数超过 batch_size，则切分为多个批次并发处理。
            else:
                # 按 batch_size 切分 memories。
                batches = self._split_batches(memories=memories, batch_size=batch_size)

                # all_success 初始为 True。
                # 只要任一批失败，就改为 False。
                all_success = True

                # 统计失败批次数。
                failed_batches = 0

                # 局部导入 as_completed，只有多批并发路径需要。
                from concurrent.futures import as_completed

                # 局部导入 ContextThreadPoolExecutor，保持请求上下文在线程池中传播。
                from memos.context.context import ContextThreadPoolExecutor

                # 每个批次一个 worker，同时并发增强。
                with ContextThreadPoolExecutor(max_workers=len(batches)) as executor:
                    # 提交所有批次增强任务。
                    # future_map 记录 future 对应的批次信息，方便后续定位。
                    future_map = {
                        executor.submit(
                            self._process_enhancement_batch, bi, query_history, texts, retries
                        ): (bi, s, e)
                        for bi, (s, e, texts) in enumerate(batches)
                    }

                    # 收集所有批次增强结果。
                    enhanced_memories = []

                    # 按完成顺序遍历 future。
                    # 注意这可能导致 enhanced_memories 的顺序与原始批次顺序不同。
                    for fut in as_completed(future_map):
                        # 取出该 future 对应的批次元信息。
                        _bi, _s, _e = future_map[fut]

                        # 获取批次结果。
                        # 如果批次内部失败，它通常会返回原 memories 和 ok=False。
                        batch_memories, ok = fut.result()

                        # 合并当前批次结果。
                        enhanced_memories.extend(batch_memories)

                        # 如果当前批次失败，则更新总成功标记和失败批次数。
                        if not ok:
                            all_success = False
                            failed_batches += 1

                # 记录多批增强汇总。
                logger.info(
                    "[Enhance] multi-batch done | batches=%s | enhanced=%s | failed_batches=%s | success=%s",
                    len(batches),
                    len(enhanced_memories),
                    failed_batches,
                    all_success,
                )

        # 捕获整个增强流程中的致命异常。
        # 这种异常可能发生在分批、线程池、future.result 等外层步骤。
        except Exception as e:
            # 记录错误。
            logger.error("[Enhance] fatal error: %s", e, exc_info=True)

            # 标记整体失败。
            all_success = False

            # 回退到原始 memories，避免增强失败导致上游没有可用数据。
            enhanced_memories = memories

        # 如果最终增强结果为空，则记录错误并返回空列表。
        # 这里没有回退到原始 memories，说明作者认为“LLM 明确产出空列表”是严重异常。
        if len(enhanced_memories) == 0:
            enhanced_memories = []
            logger.error("[Enhance] fatal error: enhanced_memories is empty", exc_info=True)

        # 返回增强后的 memories 和整体成功标记。
        return enhanced_memories, all_success
