"""
Search handler for memory search functionality (Class-based version).

This module provides a class-based implementation of search handlers,
using dependency injection for better modularity and testability.
"""

# copy 用于复制请求对象。
# 搜索流程会临时扩大 top_k、补默认 relativity 等，因此不能直接修改调用方传入的原始 search_req。
import copy

# math 用于 MMR 去重中的指数惩罚和 TF-IDF 余弦相似度计算。
import math

# os 用于读取环境变量。
# 本文件中主要控制 dream context recall 是否启用，以及 recall top_k。
import os

# suppress 用于简化环境变量解析逻辑。
# 当 int 转换失败时，直接吞掉异常并返回默认值。
from contextlib import suppress

# Any 用于给复杂字典、hook 结果和 memory 结构做宽泛类型标注。
from typing import Any

# BaseHandler 是所有 API handler 的公共基类。
# HandlerDependencies 是依赖注入容器，里面提供 naive_mem_cube、searcher、scheduler 等运行依赖。
from memos.api.handlers.base_handler import BaseHandler, HandlerDependencies

# rerank_knowledge_mem 用于搜索结果的知识记忆重排。
# 它会根据 query 对 text_mem 做进一步排序，并处理文件记忆比例。
from memos.api.handlers.formatters_handler import rerank_knowledge_mem

# APISearchRequest 是搜索接口请求模型。
# SearchResponse 是搜索接口统一响应模型。
from memos.api.product_models import APISearchRequest, SearchResponse

# CONTEXT_MEMORY_TYPE 表示 dream/contextualization 相关的上下文记忆类型。
# 环境变量开启 context recall 时，会用它限制图数据库召回范围。
from memos.dream.contextualization import CONTEXT_MEMORY_TYPE

# 获取模块级 logger。
# 虽然类内部主要使用 self.logger，这里仍有模块级 logger 给辅助逻辑使用。
from memos.log import get_logger

# cosine_similarity_matrix 用于根据 embedding 计算候选记忆之间的相似度矩阵。
# sim 去重和 mmr 去重都依赖这个矩阵判断内容是否过于相似。
from memos.memories.textual.tree_text_memory.retrieve.retrieve_utils import (
    cosine_similarity_matrix,
)

# CompositeCubeView 表示多个 cube 的组合视图。
# 当 readable_cube_ids 有多个目标 cube 时，搜索会通过它聚合多 cube 结果。
from memos.multi_mem_cube.composite_cube import CompositeCubeView

# SingleCubeView 表示单个 cube 的操作视图。
# 当只搜索一个 cube 时，直接用它执行 search_memories。
from memos.multi_mem_cube.single_cube import SingleCubeView

# MemCubeView 是 cube view 的抽象类型。
# _build_cube_view 根据 cube 数量返回 SingleCubeView 或 CompositeCubeView。
from memos.multi_mem_cube.views import MemCubeView

# H 定义 hook 名称常量。
# 本文件中在搜索结果生成、rerank 后、上下文渲染阶段触发 hook。
from memos.plugins.hook_defs import H

# hookable 把 handler 方法注册为可被插件系统拦截的业务入口。
# trigger_hook 则在搜索流程中主动触发某个 hook 点。
from memos.plugins.hooks import hookable, trigger_hook


# 当前模块 logger。
# 这里用 __name__ 让日志能定位到 search handler 模块。
logger = get_logger(__name__)

# 控制是否启用 dream context recall 的环境变量名。
_ENV_CONTEXT_RECALL = "MEMOS_DREAM_CONTEXT_RECALL"

# 控制 context recall 返回数量的环境变量名。
_ENV_CONTEXT_RECALL_TOP_K = "MEMOS_DREAM_CONTEXT_RECALL_TOP_K"

# context recall 默认召回数量。
# 当环境变量未设置或解析失败时使用该值。
_DEFAULT_CONTEXT_RECALL_TOP_K = 2


# 读取布尔型环境变量。
# 该函数把 "0"、"false"、"no"、"off" 统一视为关闭，其他值视为开启。
def _env_enabled(name: str, default: str = "off") -> bool:
    # os.getenv 读取环境变量；strip/lower 让判断不受空格和大小写影响。
    return os.getenv(name, default).strip().lower() not in {"0", "false", "no", "off"}


# 读取整型环境变量。
# 如果环境变量不存在、类型异常或无法转成 int，就返回默认值。
def _env_int(name: str, default: int) -> int:
    # suppress 会吞掉 TypeError 和 ValueError。
    # 这样非法配置不会让搜索接口直接失败。
    with suppress(TypeError, ValueError):
        # 先把默认值转成字符串传给 getenv，再整体 int 转换。
        return int(os.getenv(name, str(default)))

    # 解析失败时返回调用方给定的默认值。
    return default


# SearchHandler 负责处理“记忆搜索”接口。
# 它并不直接实现底层检索，而是组织请求复制、cube view 构建、hook、去重、重排和响应封装。
class SearchHandler(BaseHandler):
    """
    Handler for memory search operations.

    Provides fast, fine-grained, and mixture-based search modes.
    """

    # 初始化搜索 handler。
    # 依赖通过 HandlerDependencies 注入，便于测试和替换底层组件。
    def __init__(self, dependencies: HandlerDependencies):
        """
        Initialize search handler.

        Args:
            dependencies: HandlerDependencies instance
        """
        # 调用 BaseHandler 初始化公共依赖属性，例如 self.searcher、self.mem_scheduler、self.logger 等。
        super().__init__(dependencies)

        # 校验搜索流程必需依赖。
        # 缺依赖时应在 handler 初始化阶段尽早暴露，而不是请求处理中途失败。
        self._validate_dependencies(
            "naive_mem_cube", "mem_scheduler", "searcher", "deepsearch_agent"
        )

    # 将搜索入口注册到插件系统的 "search" hook。
    # 外部插件可以围绕该方法做前置、后置或替换逻辑。
    @hookable("search")
    # 搜索接口主入口。
    # 这里负责把 APISearchRequest 转换成最终 SearchResponse。
    def handle_search_memories(self, search_req: APISearchRequest) -> SearchResponse:
        """
        Main handler for search memories endpoint.

        Orchestrates the search process based on the requested search mode,
        supporting text memory searches.

        Args:
            search_req: Search request containing query and parameters

        Returns:
            SearchResponse with formatted results
        """
        # 记录原始搜索请求。
        # 这有助于排查 query、top_k、mode、dedup、filter 等参数如何影响最终结果。
        self.logger.info(f"[SearchHandler] Search Req is: {search_req}")

        # Use deepcopy to avoid modifying the original request object
        # 深拷贝一份本地请求对象，后续所有内部改写都作用在副本上。
        # 例如 dedup 时会扩大 top_k，如果直接改原对象会影响调用方语义。
        search_req_local = copy.deepcopy(search_req)

        # Expand top_k for deduplication (5x to ensure enough candidates)
        # 如果启用 sim 或 mmr 去重，需要先多召回一些候选。
        # 因为去重会删掉相似结果，如果只取原始 top_k，最终可能不够数量。
        if search_req_local.dedup in ("sim", "mmr"):
            # 当前实现扩大为 3 倍。
            # 注释里提到 5x，但实际代码是 *3，阅读时应以代码为准。
            search_req_local.top_k = search_req_local.top_k * 3

        # Search and deduplicate
        # 根据请求解析目标 cube，并构建单 cube 或多 cube 的搜索视图。
        cube_view = self._build_cube_view(search_req_local)

        # 通过 cube_view 执行底层搜索。
        # 单 cube 会调用 SingleCubeView.search_memories，多 cube 会由 CompositeCubeView 聚合。
        results = cube_view.search_memories(search_req_local)

        # 触发“搜索结果生成后”的 hook。
        # 插件可以在 rerank、去重前查看或修改原始搜索结果。
        hooked_results = trigger_hook(
            H.SEARCH_MEMORY_RESULTS,
            handler=self,
            search_req=search_req_local,
            results=results,
        )

        # 如果 hook 返回了新的结果，则用 hook 结果替换当前 results。
        if hooked_results is not None:
            results = hooked_results

        # relativity 为空时统一补成 0。
        # 这样后续阈值过滤函数可以用 <= 0 判断是否跳过过滤。
        if not search_req_local.relativity:
            search_req_local.relativity = 0

        # 记录最终使用的相关度过滤阈值。
        self.logger.info(f"[SearchHandler] Relativity filter: {search_req_local.relativity}")

        # 根据 relativity 阈值过滤 text_mem 和 pref_mem 中的 memories。
        # relativity <= 0 时函数会原样返回。
        results = self._apply_relativity_threshold(results, search_req_local.relativity)

        # 如果使用 sim 去重，则基于 embedding 相似度做严格相似项过滤。
        if search_req_local.dedup == "sim":
            # 注意这里传入的是原始 search_req.top_k，而不是扩大后的 search_req_local.top_k。
            # 这样最终每个 bucket 最多保留用户真正请求的数量。
            results = self._dedup_text_memories(results, search_req.top_k)

            # 去重计算后移除 embedding，避免响应体太大或暴露向量细节。
            self._strip_embeddings(results)

        # 如果使用 mmr 去重，则用 relevance/diversity 平衡选择结果。
        elif search_req_local.dedup == "mmr":
            # 偏好记忆可以有单独 top_k；没有配置时默认取 6。
            pref_top_k = getattr(search_req_local, "pref_top_k", 6)

            # 对 text_mem 和 pref_mem 一起做 MMR 风格去重。
            # 这里同样用原始 search_req.top_k 控制最终文本记忆数量。
            results = self._mmr_dedup_text_memories(results, search_req.top_k, pref_top_k)

            # MMR 完成后清空响应中的 embedding。
            self._strip_embeddings(results)

        # 取出文本记忆 bucket，准备做知识记忆重排。
        text_mem = results["text_mem"]

        # 对 text_mem 做 rerank。
        # reranker 会结合 query、记忆内容和文件记忆比例，重新排序或截断结果。
        results["text_mem"] = rerank_knowledge_mem(
            self.reranker,
            query=search_req.query,
            text_mem=text_mem,
            top_k=search_req_local.top_k,
            file_mem_proportion=0.5,
        )

        # 触发 rerank 后 hook。
        # 插件可以在最终上下文渲染前调整排序后的结果。
        hooked_results = trigger_hook(
            H.SEARCH_RESULTS_AFTER_RERANK,
            handler=self,
            search_req=search_req_local,
            results=results,
        )

        # 如果 hook 返回新结果，则替换当前结果。
        if hooked_results is not None:
            results = hooked_results

        # 触发上下文渲染阶段 hook。
        # 这个 hook 通常用于把搜索结果转换成更适合下游 LLM 使用的上下文结构。
        hooked_results = trigger_hook(
            H.SEARCH_CONTEXT_RENDER,
            handler=self,
            search_req=search_req_local,
            results=results,
        )

        # 如果上下文渲染 hook 返回结果，则采用它。
        if hooked_results is not None:
            results = hooked_results

        # 记录最终搜索结果。
        # len(results) 是顶层 dict key 数量，不一定等于 memory 数量，这一点调试时需要注意。
        self.logger.info(
            f"[SearchHandler] Final search results: count={len(results)} results={results}"
        )

        # 封装统一 API 响应。
        return SearchResponse(
            message="Search completed successfully",
            data=results,
        )

    # 将 dream context recall 的结果合并进现有搜索结果。
    # 当前主流程 handle_search_memories 中没有调用该方法，可能由 hook 或后续版本接入。
    def _merge_context_recall(
        self, *, results: dict[str, Any], search_req: APISearchRequest
    ) -> None:
        # 环境变量未开启时，直接跳过 context recall。
        if not _env_enabled(_ENV_CONTEXT_RECALL, "off"):
            return

        # 读取 context recall top_k，并保证最小值不低于 0。
        top_k = max(0, _env_int(_ENV_CONTEXT_RECALL_TOP_K, _DEFAULT_CONTEXT_RECALL_TOP_K))

        # top_k 为 0 时没有召回意义，直接返回。
        if top_k <= 0:
            return

        # 从图数据库中召回 context bucket。
        context_buckets = self._recall_context_buckets(search_req=search_req, top_k=top_k)

        # 没有召回任何 context 时不修改结果。
        if not context_buckets:
            return

        # 将 context bucket 追加到 text_mem。
        # setdefault 确保 results 中没有 text_mem 时也能创建列表。
        results.setdefault("text_mem", []).extend(context_buckets)

    # 根据 query embedding 从每个可读 cube 中召回上下文记忆 bucket。
    # 返回结构与 text_mem bucket 类似，便于合并到普通搜索结果。
    def _recall_context_buckets(
        self, *, search_req: APISearchRequest, top_k: int
    ) -> list[dict[str, Any]]:
        # 优先使用 handler 直接持有的 graph_db；没有时尝试从 searcher 上取 graph_store。
        graph_db = self.graph_db or getattr(self.searcher, "graph_store", None)

        # 优先使用 handler 直接持有的 embedder；没有时尝试从 searcher 上取 embedder。
        embedder = self.embedder or getattr(self.searcher, "embedder", None)

        # 缺少图数据库或 embedder 时无法做 embedding 召回。
        if graph_db is None or embedder is None:
            self.logger.info(
                "[SearchHandler] Context recall skipped: graph_db or embedder unavailable."
            )
            return []

        try:
            # 将搜索 query 转成 embedding，用于图数据库向量检索。
            query_embedding = embedder.embed([search_req.query])[0]

        # embedding 生成失败时跳过 context recall，不影响普通搜索主流程。
        except Exception:
            self.logger.warning("[SearchHandler] Context recall embedding failed.", exc_info=True)
            return []

        # 收集每个 cube 的 context bucket。
        buckets: list[dict[str, Any]] = []

        # context recall 需要针对每个可读 cube 分别检索。
        for cube_id in self._resolve_cube_ids(search_req):
            try:
                # 在当前 cube 命名空间下做向量检索。
                # scope 限定为 CONTEXT_MEMORY_TYPE，status 限定为 activated。
                hits = graph_db.search_by_embedding(
                    query_embedding,
                    top_k=top_k,
                    scope=CONTEXT_MEMORY_TYPE,
                    status="activated",
                    user_name=cube_id,
                    return_fields=[
                        "memory",
                        "key",
                        "created_at",
                        "updated_at",
                        "source",
                        "internal_info",
                    ],
                )

            # 单个 cube 召回失败不影响其他 cube。
            except Exception:
                self.logger.warning(
                    "[SearchHandler] Context recall search failed for cube=%s.",
                    cube_id,
                    exc_info=True,
                )
                continue

            # 将图数据库 hit 转成前端/下游统一 memory 结构。
            # 只保留包含 memory 内容的 hit。
            memories = [self._format_context_hit(hit) for hit in hits or [] if hit.get("memory")]

            # 如果当前 cube 没有有效 context memory，就不创建 bucket。
            if not memories:
                continue

            # 构造一个 text_mem 风格的 bucket。
            buckets.append(
                {
                    "cube_id": cube_id,
                    "memories": memories,
                    "total_nodes": len(memories),
                }
            )

        # 返回所有 cube 的 context bucket。
        return buckets

    # 把图数据库返回的 context hit 格式化成普通 memory dict。
    # 静态方法说明它不依赖 handler 实例状态。
    @staticmethod
    def _format_context_hit(hit: dict[str, Any]) -> dict[str, Any]:
        # context_id 从 hit.id 转成字符串。
        # 缺失时使用空字符串，后面 ref_id 会退化为 [context]。
        context_id = str(hit.get("id", ""))

        # score 表示向量检索相关度。
        # 转 float 可以统一后续 relativity/score 字段类型。
        score = float(hit.get("score", 0.0) or 0.0)

        # 构造 metadata，尽量对齐普通 TextualMemoryItem 的响应结构。
        metadata = {
            "id": context_id,
            "memory": hit.get("memory", ""),
            "memory_type": CONTEXT_MEMORY_TYPE,
            "source": hit.get("source") or "dream",
            "key": hit.get("key", ""),
            "relativity": score,
            "score": score,
            "embedding": [],
            "sources": [],
            "usage": [],
            "ref_id": f"[{context_id.split('-')[0]}]" if context_id else "[context]",
        }

        # 补充可选字段。
        # 只有 hit 中存在且非 None 时才写入 metadata，避免输出大量空字段。
        for field in ("created_at", "updated_at", "internal_info"):
            if hit.get(field) is not None:
                metadata[field] = hit[field]

        # 返回统一 memory dict。
        return {
            "id": context_id,
            "memory": hit.get("memory", ""),
            "metadata": metadata,
            "ref_id": metadata["ref_id"],
        }

    # 对搜索结果应用 relativity 阈值过滤。
    # 它会原地修改 results 中 text_mem/pref_mem 的 bucket 内容，并返回同一个 results。
    @staticmethod
    def _apply_relativity_threshold(results: dict[str, Any], relativity: float) -> dict[str, Any]:
        # relativity <= 0 表示不启用过滤。
        if relativity <= 0:
            return results

        # 当前只过滤文本记忆和偏好记忆。
        # 其他类型如 tool_mem、skill_mem 不参与该阈值过滤。
        for key in ("text_mem", "pref_mem"):
            # 取出对应 bucket 列表。
            buckets = results.get(key)

            # 如果结构不是 list，说明没有该类结果或格式异常，跳过。
            if not isinstance(buckets, list):
                continue

            # 遍历每个 bucket。
            for bucket in buckets:
                # bucket 中的 memories 才是真正的记忆列表。
                memories = bucket.get("memories")

                # memories 不是 list 时跳过该 bucket。
                if not isinstance(memories, list):
                    continue

                # filtered 保存通过阈值检查的记忆。
                filtered: list[dict[str, Any]] = []

                # 逐条检查 memory。
                for mem in memories:
                    # 只处理 dict 结构。
                    if not isinstance(mem, dict):
                        continue

                    # metadata 中通常会包含 relativity。
                    meta = mem.get("metadata", {})

                    # 如果 metadata 是 dict，则读取 relativity；否则默认认为相关度为 1.0。
                    score = meta.get("relativity", 1.0) if isinstance(meta, dict) else 1.0

                    try:
                        # 将 score 转为 float，便于与阈值比较。
                        score_val = float(score) if score is not None else 1.0

                    # score 不可转换时默认保留。
                    except (TypeError, ValueError):
                        score_val = 1.0

                    # 只有相关度达到阈值的记忆才保留。
                    if score_val >= relativity:
                        filtered.append(mem)

                # 用过滤后的列表替换原 memories。
                bucket["memories"] = filtered

                # 如果 bucket 里维护 total_nodes，也同步更新数量。
                if "total_nodes" in bucket:
                    bucket["total_nodes"] = len(filtered)

        # 返回修改后的结果。
        return results

    # 基于 embedding 相似度的文本记忆去重。
    # 它只处理 results["text_mem"]，不处理 pref_mem。
    def _dedup_text_memories(self, results: dict[str, Any], target_top_k: int) -> dict[str, Any]:
        # 取出文本记忆 buckets。
        buckets = results.get("text_mem", [])

        # 没有文本记忆时无需去重。
        if not buckets:
            return results

        # 将所有 bucket 内的 memory 拉平成一个列表。
        # tuple 结构为：原 bucket 下标、memory dict、原始相关度分数。
        flat: list[tuple[int, dict[str, Any], float]] = []

        # 遍历 bucket，保留 bucket_idx 是为了去重后还能放回原 bucket。
        for bucket_idx, bucket in enumerate(buckets):
            # 遍历 bucket 内每条 memory。
            for mem in bucket.get("memories", []):
                # relativity 用作候选排序分数，默认 0。
                score = mem.get("metadata", {}).get("relativity", 0.0)

                # 添加到扁平候选列表。
                flat.append((bucket_idx, mem, score))

        # 0 或 1 条候选不需要去重。
        if len(flat) <= 1:
            return results

        # 提取或补算所有候选 memory 的 embedding。
        embeddings = self._extract_embeddings([mem for _, mem, _ in flat])

        # 根据 embedding 计算候选两两相似度矩阵。
        similarity_matrix = cosine_similarity_matrix(embeddings)

        # 建立 bucket -> flat index 列表的映射。
        # 该结构当前后续没有直接用于选择，但保留有助于未来按 bucket 处理。
        indices_by_bucket: dict[int, list[int]] = {i: [] for i in range(len(buckets))}

        # 把每个 flat index 归到所属 bucket。
        for flat_index, (bucket_idx, _, _) in enumerate(flat):
            indices_by_bucket[bucket_idx].append(flat_index)

        # selected_global 保存跨 bucket 已选择的 flat index。
        # 跨 bucket 共享它，可以避免不同 cube 或 bucket 里返回过于相似的文本。
        selected_global: list[int] = []

        # selected_by_bucket 保存每个 bucket 最终选择的 flat index。
        selected_by_bucket: dict[int, list[int]] = {i: [] for i in range(len(buckets))}

        # 按相关度从高到低遍历候选。
        # 先看高分候选，再用相似度阈值过滤重复内容。
        ordered_indices = sorted(range(len(flat)), key=lambda idx: flat[idx][2], reverse=True)

        # 遍历候选下标。
        for idx in ordered_indices:
            # 找到候选所属 bucket。
            bucket_idx = flat[idx][0]

            # 如果该 bucket 已经达到目标 top_k，就不再往里面加。
            if len(selected_by_bucket[bucket_idx]) >= target_top_k:
                continue

            # Use 0.92 threshold strictly
            # 只有与已选结果相似度都不超过 0.92 时，才认为它足够不重复。
            if self._is_unrelated(idx, selected_global, similarity_matrix, 0.92):
                # 记录到所属 bucket 的选择结果。
                selected_by_bucket[bucket_idx].append(idx)

                # 同时记录到全局选择集合，用于后续候选跨 bucket 去重。
                selected_global.append(idx)

        # Removed the 'filling' logic that was pulling back similar items.
        # Now it will only return items that truly pass the 0.92 threshold,
        # up to target_top_k.

        # 将每个 bucket 的 memories 替换为选中的去重结果。
        for bucket_idx, bucket in enumerate(buckets):
            # 获取当前 bucket 被选中的 flat index。
            selected_indices = selected_by_bucket.get(bucket_idx, [])

            # 用 flat index 找回 memory dict。
            bucket["memories"] = [flat[i][1] for i in selected_indices]

        # 返回修改后的 results。
        return results

    # 基于 MMR 思路的去重。
    # 相比 sim 去重，它不仅看相似度阈值，还在相关度和多样性之间做平衡。
    def _mmr_dedup_text_memories(
        self, results: dict[str, Any], text_top_k: int, pref_top_k: int = 6
    ) -> dict[str, Any]:
        """
        MMR-based deduplication with progressive penalty for high similarity.

        Performs deduplication on both text_mem and preference memories together.
        Other memory types (tool_mem, etc.) are not modified.

        Args:
            results: Search results containing text_mem and preference buckets
            text_top_k: Target number of text memories to return per bucket
            pref_top_k: Target number of preference memories to return per bucket

        Algorithm:
        1. Prefill top 5 by relevance
        2. MMR selection: balance relevance vs diversity
        3. Re-sort by original relevance for better generation quality
        """
        # 取出文本记忆 buckets。
        text_buckets = results.get("text_mem", [])

        # 取出偏好记忆 buckets。
        pref_buckets = results.get("pref_mem", [])

        # Early return if no memories to deduplicate
        # 如果两类 bucket 都不存在，就没有去重对象。
        if not text_buckets and not pref_buckets:
            return results

        # Flatten all memories with their type and scores
        # flat structure: (memory_type, bucket_idx, mem, score)
        # 扁平化后才能跨 text/pref 统一计算 embedding 相似度和 MMR 分数。
        flat: list[tuple[str, int, dict[str, Any], float]] = []

        # Flatten text memories
        # 先处理 text_mem，并标记类型为 "text"。
        for bucket_idx, bucket in enumerate(text_buckets):
            for mem in bucket.get("memories", []):
                # 文本记忆以 metadata.relativity 作为相关度。
                score = mem.get("metadata", {}).get("relativity", 0.0)

                # score 统一转 float，None 退化为 0。
                flat.append(("text", bucket_idx, mem, float(score) if score is not None else 0.0))

        # Flatten preference memories
        # 再处理 pref_mem，并标记类型为 "preference"。
        for bucket_idx, bucket in enumerate(pref_buckets):
            for mem in bucket.get("memories", []):
                # 偏好记忆可能使用 score，也可能使用 relativity。
                meta = mem.get("metadata", {})

                # metadata 是 dict 时优先取 score，否则取 relativity。
                if isinstance(meta, dict):
                    score = meta.get("score", meta.get("relativity", 0.0))
                else:
                    score = 0.0

                # 添加偏好记忆候选。
                flat.append(
                    ("preference", bucket_idx, mem, float(score) if score is not None else 0.0)
                )

        # 候选不足两条时不需要做 MMR。
        if len(flat) <= 1:
            return results

        # 统计每种类型候选总数。
        total_by_type: dict[str, int] = {"text": 0, "preference": 0}

        # 统计每种类型已有 embedding 的数量。
        existing_by_type: dict[str, int] = {"text": 0, "preference": 0}

        # 统计每种类型缺失 embedding 的数量。
        missing_by_type: dict[str, int] = {"text": 0, "preference": 0}

        # 记录缺失 embedding 的 flat index。
        missing_indices: list[int] = []

        # 扫描所有扁平候选，统计 embedding 元数据情况。
        for idx, (mem_type, _, mem, _) in enumerate(flat):
            # 理论上 mem_type 只有 text/preference。
            # 这里做扩展性保护，避免未来加入新类型时 KeyError。
            if mem_type not in total_by_type:
                total_by_type[mem_type] = 0
                existing_by_type[mem_type] = 0
                missing_by_type[mem_type] = 0

            # 当前类型总数加一。
            total_by_type[mem_type] += 1

            # 尝试从 metadata 中取 embedding。
            embedding = mem.get("metadata", {}).get("embedding")

            # 有 embedding 则计入 existing。
            if embedding:
                existing_by_type[mem_type] += 1

            # 没有 embedding 则计入 missing，并记录下标。
            else:
                missing_by_type[mem_type] += 1
                missing_indices.append(idx)

        # 记录 embedding 元数据扫描结果。
        # 这能帮助判断 MMR 是否需要大量临时补算 embedding。
        self.logger.info(
            "[SearchHandler] MMR embedding metadata scan: total=%s total_by_type=%s existing_by_type=%s missing_by_type=%s",
            len(flat),
            total_by_type,
            existing_by_type,
            missing_by_type,
        )

        # 如果存在缺失 embedding，记录 warning。
        # 这不是错误，因为后续 _extract_embeddings 会自动补算。
        if missing_indices:
            self.logger.warning(
                "[SearchHandler] MMR embedding metadata missing; will compute missing embeddings: missing_total=%s",
                len(missing_indices),
            )

        # Get or compute embeddings
        # 提取已有 embedding，并对缺失项调用 embedder 补算。
        embeddings = self._extract_embeddings([mem for _, _, mem, _ in flat])

        # Compute similarity matrix using NumPy-optimized method
        # Returns numpy array but compatible with list[i][j] indexing
        # 计算两两 embedding 相似度矩阵，后续用于多样性惩罚和高相似跳过。
        similarity_matrix = cosine_similarity_matrix(embeddings)

        # Initialize selection tracking for both text and preference
        # 分别记录每个文本 bucket 包含哪些 flat index。
        text_indices_by_bucket: dict[int, list[int]] = {i: [] for i in range(len(text_buckets))}

        # 分别记录每个偏好 bucket 包含哪些 flat index。
        pref_indices_by_bucket: dict[int, list[int]] = {i: [] for i in range(len(pref_buckets))}

        # 根据 flat 中的类型和 bucket_idx 填充上述映射。
        for flat_index, (mem_type, bucket_idx, _, _) in enumerate(flat):
            # 文本候选归入 text_indices_by_bucket。
            if mem_type == "text":
                text_indices_by_bucket[bucket_idx].append(flat_index)

            # 偏好候选归入 pref_indices_by_bucket。
            elif mem_type == "preference":
                pref_indices_by_bucket[bucket_idx].append(flat_index)

        # selected_global 保存所有已选候选下标，用于全局相似度判断。
        selected_global: list[int] = []

        # 每个文本 bucket 已选的候选下标。
        text_selected_by_bucket: dict[int, list[int]] = {i: [] for i in range(len(text_buckets))}

        # 每个偏好 bucket 已选的候选下标。
        pref_selected_by_bucket: dict[int, list[int]] = {i: [] for i in range(len(pref_buckets))}

        # Track exact text content to avoid duplicates
        # 用原始文本字符串去重，防止完全相同内容重复进入结果。
        selected_texts: set[str] = set()

        # Phase 1: Prefill top N by relevance
        # Use the smaller of text_top_k and pref_top_k for prefill count
        # 第一阶段先按相关度预填少量高分结果，为 MMR 提供初始集合。
        prefill_top_n = min(2, text_top_k, pref_top_k) if pref_buckets else min(2, text_top_k)

        # 按候选原始相关度降序排序。
        ordered_by_relevance = sorted(range(len(flat)), key=lambda idx: flat[idx][3], reverse=True)

        # 遍历所有候选，但最多预填 prefill_top_n 条。
        for idx in ordered_by_relevance[: len(flat)]:
            # 如果预填数量达到上限，则结束第一阶段。
            if len(selected_global) >= prefill_top_n:
                break

            # 取出当前候选的类型、bucket、memory。
            mem_type, bucket_idx, mem, _ = flat[idx]

            # Skip if exact text already exists in selected set
            # 去掉首尾空格后做精确文本重复判断。
            mem_text = mem.get("memory", "").strip()

            # 已选过完全相同文本则跳过。
            if mem_text in selected_texts:
                continue

            # Skip if highly similar (Dice + TF-IDF + 2-gram combined, with embedding filter)
            # 在 embedding 预过滤基础上，再用文本相似度组合算法判断是否高度相似。
            if SearchHandler._is_text_highly_similar_optimized(
                idx, mem_text, selected_global, similarity_matrix, flat, threshold=0.92
            ):
                continue

            # Check bucket capacity with correct top_k for each type
            # 文本记忆必须没有超过 text_top_k 才能加入。
            if mem_type == "text" and len(text_selected_by_bucket[bucket_idx]) < text_top_k:
                selected_global.append(idx)
                text_selected_by_bucket[bucket_idx].append(idx)
                selected_texts.add(mem_text)

            # 偏好记忆必须没有超过 pref_top_k 才能加入。
            elif mem_type == "preference" and len(pref_selected_by_bucket[bucket_idx]) < pref_top_k:
                selected_global.append(idx)
                pref_selected_by_bucket[bucket_idx].append(idx)
                selected_texts.add(mem_text)

        # Phase 2: MMR selection for remaining slots
        # 第二阶段用 MMR 分数继续选择剩余候选。
        lambda_relevance = 0.8

        # 相似度超过该阈值后，开始施加指数惩罚。
        similarity_threshold = 0.9  # Start exponential penalty from 0.9 (lowered from 0.9)

        # 指数惩罚系数。
        # 越大表示对高相似候选惩罚越重。
        alpha_exponential = 10.0  # Exponential penalty coefficient

        # remaining 是还没有被选中的所有候选下标。
        remaining = set(range(len(flat))) - set(selected_global)

        # 持续选择，直到没有可选候选或所有 bucket 满额。
        while remaining:
            # 当前轮最佳候选下标。
            best_idx: int | None = None

            # 当前轮最佳 MMR 分数。
            best_mmr: float | None = None

            # 遍历所有剩余候选，寻找本轮最优。
            for idx in remaining:
                # 取出候选信息。
                mem_type, bucket_idx, mem, _ = flat[idx]

                # Check bucket capacity with correct top_k for each type
                # 如果候选所属 bucket 已经满额，则它不再参与竞争。
                if (
                    mem_type == "text" and len(text_selected_by_bucket[bucket_idx]) >= text_top_k
                ) or (
                    mem_type == "preference"
                    and len(pref_selected_by_bucket[bucket_idx]) >= pref_top_k
                ):
                    continue

                # Check if exact text already exists - if so, skip this candidate entirely
                # 完全相同的文本直接跳过，不让它参与 MMR 竞争。
                mem_text = mem.get("memory", "").strip()

                if mem_text in selected_texts:
                    continue  # Skip duplicate text, don't participate in MMR competition

                # Skip if highly similar (Dice + TF-IDF + 2-gram combined, with embedding filter)
                # 高度相似文本也直接跳过，避免 MMR 因相关度高而重新选回重复项。
                if SearchHandler._is_text_highly_similar_optimized(
                    idx, mem_text, selected_global, similarity_matrix, flat, threshold=0.92
                ):
                    continue  # Skip highly similar text, don't participate in MMR competition

                # relevance 使用原始搜索相关度。
                relevance = flat[idx][3]

                # max_sim 表示该候选与已选集合中最相似项的相似度。
                max_sim = (
                    0.0
                    if not selected_global
                    else max(similarity_matrix[idx][j] for j in selected_global)
                )

                # Exponential penalty for similarity > 0.80
                # 相似度超过阈值时，用指数放大惩罚项。
                if max_sim > similarity_threshold:
                    penalty_multiplier = math.exp(
                        alpha_exponential * (max_sim - similarity_threshold)
                    )
                    diversity = max_sim * penalty_multiplier

                # 相似度未超过阈值时，直接用 max_sim 作为多样性惩罚。
                else:
                    diversity = max_sim

                # MMR 分数 = 相关度收益 - 多样性惩罚。
                # lambda_relevance 越大越偏向相关度，越小越偏向多样性。
                mmr_score = lambda_relevance * relevance - (1.0 - lambda_relevance) * diversity

                # 更新本轮最佳候选。
                if best_mmr is None or mmr_score > best_mmr:
                    best_mmr = mmr_score
                    best_idx = idx

            # 如果没有候选能通过容量/重复/相似度检查，则结束。
            if best_idx is None:
                break

            # 取出本轮最佳候选的信息。
            mem_type, bucket_idx, mem, _ = flat[best_idx]

            # Add to selected set and track text
            # 标记该候选已被选中。
            mem_text = mem.get("memory", "").strip()

            # 加入全局已选列表。
            selected_global.append(best_idx)

            # 记录精确文本，后续避免重复。
            selected_texts.add(mem_text)

            # 按类型加入对应 bucket 的已选列表。
            if mem_type == "text":
                text_selected_by_bucket[bucket_idx].append(best_idx)
            elif mem_type == "preference":
                pref_selected_by_bucket[bucket_idx].append(best_idx)

            # 从 remaining 中删除已选候选。
            remaining.remove(best_idx)

            # Early termination: all buckets are full
            # 检查所有文本 bucket 是否已经达到 min(top_k, 该 bucket 原候选数)。
            text_all_full = all(
                len(text_selected_by_bucket[b_idx]) >= min(text_top_k, len(bucket_indices))
                for b_idx, bucket_indices in text_indices_by_bucket.items()
            )

            # 检查所有偏好 bucket 是否已经达到 min(pref_top_k, 该 bucket 原候选数)。
            pref_all_full = all(
                len(pref_selected_by_bucket[b_idx]) >= min(pref_top_k, len(bucket_indices))
                for b_idx, bucket_indices in pref_indices_by_bucket.items()
            )

            # 如果所有 bucket 都满额，则提前结束 MMR。
            if text_all_full and pref_all_full:
                break

        # Phase 3: Re-sort by original relevance and fill back to buckets
        # 第三阶段：把选中结果按原始相关度重新排序，提升生成质量和可解释性。
        for bucket_idx, bucket in enumerate(text_buckets):
            # 获取当前文本 bucket 的已选候选。
            selected_indices = text_selected_by_bucket.get(bucket_idx, [])

            # 按原始 relevance 降序排列。
            selected_indices = sorted(selected_indices, key=lambda i: flat[i][3], reverse=True)

            # 写回 bucket memories。
            bucket["memories"] = [flat[i][2] for i in selected_indices]

        # 对偏好 bucket 做同样的回填。
        for bucket_idx, bucket in enumerate(pref_buckets):
            # 获取当前偏好 bucket 的已选候选。
            selected_indices = pref_selected_by_bucket.get(bucket_idx, [])

            # 按原始 relevance 降序排列。
            selected_indices = sorted(selected_indices, key=lambda i: flat[i][3], reverse=True)

            # 写回 bucket memories。
            bucket["memories"] = [flat[i][2] for i in selected_indices]

        # 返回已被原地修改的 results。
        return results

    # 判断某个候选是否与所有已选候选都“不太相似”。
    # 方法名 _is_unrelated 表示相似度不能超过阈值。
    @staticmethod
    def _is_unrelated(
        index: int,
        selected_indices: list[int],
        similarity_matrix: list[list[float]],
        similarity_threshold: float,
    ) -> bool:
        # 如果 selected_indices 为空，all(...) 对空集合返回 True，因此第一个候选会被允许。
        return all(similarity_matrix[index][j] <= similarity_threshold for j in selected_indices)

    # 计算某个候选与已选集合的最大 embedding 相似度。
    # 当前文件中该方法没有被主流程直接调用，但可作为 MMR 或调试辅助函数。
    @staticmethod
    def _max_similarity(
        index: int, selected_indices: list[int], similarity_matrix: list[list[float]]
    ) -> float:
        # 没有已选候选时，相似度定义为 0。
        if not selected_indices:
            return 0.0

        # 返回候选与所有已选候选中的最大相似度。
        return max(similarity_matrix[index][j] for j in selected_indices)

    # 提取 memory dict 中的 embedding。
    # 如果某些 memory 缺失 embedding，则调用 searcher.embedder 临时补算，并写回 metadata。
    def _extract_embeddings(self, memories: list[dict[str, Any]]) -> list[list[float]]:
        # embeddings 与 memories 一一对应。
        embeddings: list[list[float]] = []

        # 记录缺失 embedding 的 memory 下标。
        missing_indices: list[int] = []

        # 记录缺失 embedding 的文本内容，用于批量 embed。
        missing_documents: list[str] = []

        # 遍历所有 memory。
        for idx, mem in enumerate(memories):
            # 取 metadata。
            metadata = mem.get("metadata")

            # 如果 metadata 不是 dict，则修正为空 dict。
            # 这样后续可以安全写入 embedding。
            if not isinstance(metadata, dict):
                metadata = {}
                mem["metadata"] = metadata

            # 尝试读取已有 embedding。
            embedding = metadata.get("embedding")

            # 有 embedding 时直接加入结果。
            if embedding:
                embeddings.append(embedding)
                continue

            # 没有 embedding 时先占位空列表，保持下标对齐。
            embeddings.append([])

            # 记录缺失项下标。
            missing_indices.append(idx)

            # 记录要补算 embedding 的文本内容。
            missing_documents.append(mem.get("memory", ""))

        # 如果存在缺失项，则批量调用 embedder。
        if missing_indices:
            # 使用 self.searcher.embedder 进行 embedding 计算。
            computed = self.searcher.embedder.embed(missing_documents)

            # 将补算结果写回 embeddings 和原 memories metadata。
            for idx, embedding in zip(missing_indices, computed, strict=False):
                embeddings[idx] = embedding
                memories[idx]["metadata"]["embedding"] = embedding

        # 返回完整 embedding 列表。
        return embeddings

    # 从结果中移除 embedding。
    # 搜索响应通常不需要返回高维向量，清空可以减少响应体积。
    @staticmethod
    def _strip_embeddings(results: dict[str, Any]) -> None:
        # 遍历所有顶层结果类型，例如 text_mem、pref_mem、tool_mem 等。
        for _mem_type, mem_results in results.items():
            # 只处理列表型 bucket。
            if isinstance(mem_results, list):
                # 遍历每个 bucket。
                for bucket in mem_results:
                    # 遍历 bucket 内 memories。
                    for mem in bucket.get("memories", []):
                        # 取 metadata。
                        metadata = mem.get("metadata", {})

                        # 如果 metadata 中有 embedding，就清空为 []。
                        if "embedding" in metadata:
                            metadata["embedding"] = []

    # 计算字符集合级 Dice 相似度。
    # 注意：本类后面又定义了一次同名方法，Python 会以后面的定义覆盖前面的定义。
    @staticmethod
    def _dice_similarity(text1: str, text2: str) -> float:
        """
        Calculate Dice coefficient (character-level, fastest).

        Dice = 2 * |A ∩ B| / (|A| + |B|)
        Speed: O(n + m), ~0.05-0.1ms per comparison

        Args:
            text1: First text string
            text2: Second text string

        Returns:
            Dice similarity score between 0.0 and 1.0
        """
        # 任一文本为空时，相似度为 0。
        if not text1 or not text2:
            return 0.0

        # 转成字符集合，忽略字符重复次数。
        chars1 = set(text1)
        chars2 = set(text2)

        # 计算两个字符集合交集大小。
        intersection = len(chars1 & chars2)

        # Dice 系数公式。
        return 2 * intersection / (len(chars1) + len(chars2))

    # 计算字符 2-gram Jaccard 相似度。
    # 该定义后面也会被同名方法覆盖。
    @staticmethod
    def _bigram_similarity(text1: str, text2: str) -> float:
        """
        Calculate character-level 2-gram Jaccard similarity.

        Speed: O(n + m), ~0.1-0.2ms per comparison
        Considers local order (more strict than Dice).

        Args:
            text1: First text string
            text2: Second text string

        Returns:
            Jaccard similarity score between 0.0 and 1.0
        """
        # 任一文本为空时，相似度为 0。
        if not text1 or not text2:
            return 0.0

        # Generate 2-grams
        # 文本长度至少为 2 时生成所有连续 2 字符片段。
        # 长度不足 2 时，用原文本作为唯一 gram。
        bigrams1 = {text1[i : i + 2] for i in range(len(text1) - 1)} if len(text1) >= 2 else {text1}
        bigrams2 = {text2[i : i + 2] for i in range(len(text2) - 1)} if len(text2) >= 2 else {text2}

        # 计算 Jaccard 的交集。
        intersection = len(bigrams1 & bigrams2)

        # 计算 Jaccard 的并集。
        union = len(bigrams1 | bigrams2)

        # 并集非空时返回 intersection / union，否则返回 0。
        return intersection / union if union > 0 else 0.0

    # 计算字符级简化 TF-IDF 余弦相似度。
    # 该定义后面也会被同名方法覆盖。
    @staticmethod
    def _tfidf_similarity(text1: str, text2: str) -> float:
        """
        Calculate TF-IDF cosine similarity (character-level, no sklearn).

        Speed: O(n + m), ~0.3-0.5ms per comparison
        Considers character frequency weighting.

        Args:
            text1: First text string
            text2: Second text string

        Returns:
            Cosine similarity score between 0.0 and 1.0
        """
        # 任一文本为空时，相似度为 0。
        if not text1 or not text2:
            return 0.0

        # Counter 用于统计字符频次。
        from collections import Counter

        # Character frequency (TF)
        # 统计两个文本的字符 TF。
        tf1 = Counter(text1)
        tf2 = Counter(text2)

        # All unique characters (vocabulary)
        # 构造两段文本的字符词表。
        vocab = set(tf1.keys()) | set(tf2.keys())

        # Simple IDF: log(2 / df) where df is document frequency
        # For two documents, IDF is log(2/1)=0.693 if char appears in one doc,
        # or log(2/2)=0 if appears in both (we use log(2/1) for simplicity)
        # 这里使用简化权重：两边都出现的字符权重低，只出现在一边的字符权重高。
        idf = {char: (1.0 if char in tf1 and char in tf2 else 1.5) for char in vocab}

        # TF-IDF vectors
        # 构造两个文本的 TF-IDF 向量。
        vec1 = {char: tf1.get(char, 0) * idf[char] for char in vocab}
        vec2 = {char: tf2.get(char, 0) * idf[char] for char in vocab}

        # Cosine similarity
        # 计算点积。
        dot_product = sum(vec1[char] * vec2[char] for char in vocab)

        # 计算两个向量的 L2 范数。
        norm1 = math.sqrt(sum(v * v for v in vec1.values()))
        norm2 = math.sqrt(sum(v * v for v in vec2.values()))

        # 任一向量为零向量时，相似度为 0。
        if norm1 == 0 or norm2 == 0:
            return 0.0

        # 返回余弦相似度。
        return dot_product / (norm1 * norm2)

    # 使用 embedding 预过滤 + 多文本算法判断候选是否与已选内容高度相似。
    # 该定义后面有同名方法，会被后面的版本覆盖。
    @staticmethod
    def _is_text_highly_similar_optimized(
        candidate_idx: int,
        candidate_text: str,
        selected_global: list[int],
        similarity_matrix,
        flat: list,
        threshold: float = 0.9,
    ) -> bool:
        """
        Multi-algorithm text similarity check with embedding pre-filtering.

        Strategy:
        1. Only compare with the single highest embedding similarity item (not all 25)
        2. Only perform text comparison if embedding similarity > 0.60
        3. Use weighted combination of three algorithms:
           - Dice (40%): Fastest, character-level set similarity
           - TF-IDF (35%): Considers character frequency weighting
           - 2-gram (25%): Considers local character order

        Combined formula:
            combined_score = 0.40 * dice + 0.35 * tfidf + 0.25 * bigram

        This reduces comparisons from O(N) to O(1) per candidate, with embedding pre-filtering.
        Expected speedup: 100-200x compared to LCS approach.

        Args:
            candidate_idx: Index of candidate memory in flat list
            candidate_text: Text content of candidate memory
            selected_global: List of already selected memory indices
            similarity_matrix: Precomputed embedding similarity matrix
            flat: Flat list of all memories
            threshold: Combined similarity threshold (default 0.75)

        Returns:
            True if candidate is highly similar to any selected memory
        """
        # 没有已选内容时，不可能高度相似。
        if not selected_global:
            return False

        # Find the already-selected memory with highest embedding similarity
        # 只找到 embedding 最相似的一条已选记忆，而不是与所有已选项做文本比较。
        max_sim_idx = max(selected_global, key=lambda j: similarity_matrix[candidate_idx][j])

        # 读取最高 embedding 相似度。
        max_sim = similarity_matrix[candidate_idx][max_sim_idx]

        # If highest embedding similarity < 0.60, skip text comparison entirely
        # 这里实际阈值是 0.9；注释中的 0.60 与代码不一致，应以代码为准。
        if max_sim <= 0.9:
            return False

        # Get text of most similar memory
        # 取出最相似已选记忆的文本。
        most_similar_mem = flat[max_sim_idx][2]
        most_similar_text = most_similar_mem.get("memory", "").strip()

        # Calculate three similarity scores
        # 分别计算三种轻量文本相似度。
        dice_sim = SearchHandler._dice_similarity(candidate_text, most_similar_text)
        tfidf_sim = SearchHandler._tfidf_similarity(candidate_text, most_similar_text)
        bigram_sim = SearchHandler._bigram_similarity(candidate_text, most_similar_text)

        # Weighted combination: Dice (40%) + TF-IDF (35%) + 2-gram (25%)
        # Dice has highest weight (fastest and most reliable)
        # TF-IDF considers frequency (handles repeated characters well)
        # 2-gram considers order (catches local pattern similarity)
        # 加权组合得到最终文本相似度。
        combined_score = 0.40 * dice_sim + 0.35 * tfidf_sim + 0.25 * bigram_sim

        # 超过阈值则认为高度相似，应跳过。
        return combined_score >= threshold

    # 下面开始第二组同名静态方法。
    # 在 Python 类定义中，后面的同名定义会覆盖前面的定义。
    # 因此运行时实际使用的是这一组 _dice_similarity/_bigram_similarity/_tfidf_similarity/_is_text_highly_similar_optimized。
    @staticmethod
    def _dice_similarity(text1: str, text2: str) -> float:
        """
        Calculate Dice coefficient (character-level, fastest).

        Dice = 2 * |A ∩ B| / (|A| + |B|)
        Speed: O(n + m), ~0.05-0.1ms per comparison

        Args:
            text1: First text string
            text2: Second text string

        Returns:
            Dice similarity score between 0.0 and 1.0
        """
        # 任一文本为空时，相似度为 0。
        if not text1 or not text2:
            return 0.0

        # 使用字符集合，忽略字符重复次数。
        chars1 = set(text1)
        chars2 = set(text2)

        # 计算两个字符集合的交集大小。
        intersection = len(chars1 & chars2)

        # 返回 Dice 系数。
        return 2 * intersection / (len(chars1) + len(chars2))

    # 运行时实际生效的 2-gram 相似度函数。
    @staticmethod
    def _bigram_similarity(text1: str, text2: str) -> float:
        """
        Calculate character-level 2-gram Jaccard similarity.

        Speed: O(n + m), ~0.1-0.2ms per comparison
        Considers local order (more strict than Dice).

        Args:
            text1: First text string
            text2: Second text string

        Returns:
            Jaccard similarity score between 0.0 and 1.0
        """
        # 任一文本为空时，相似度为 0。
        if not text1 or not text2:
            return 0.0

        # Generate 2-grams
        # 文本长度至少为 2 时生成连续 2 字符片段，否则用文本自身作为 gram。
        bigrams1 = {text1[i : i + 2] for i in range(len(text1) - 1)} if len(text1) >= 2 else {text1}
        bigrams2 = {text2[i : i + 2] for i in range(len(text2) - 1)} if len(text2) >= 2 else {text2}

        # 计算交集。
        intersection = len(bigrams1 & bigrams2)

        # 计算并集。
        union = len(bigrams1 | bigrams2)

        # 返回 Jaccard 相似度。
        return intersection / union if union > 0 else 0.0

    # 运行时实际生效的字符级简化 TF-IDF 余弦相似度。
    @staticmethod
    def _tfidf_similarity(text1: str, text2: str) -> float:
        """
        Calculate TF-IDF cosine similarity (character-level, no sklearn).

        Speed: O(n + m), ~0.3-0.5ms per comparison
        Considers character frequency weighting.

        Args:
            text1: First text string
            text2: Second text string

        Returns:
            Cosine similarity score between 0.0 and 1.0
        """
        # 任一文本为空时，相似度为 0。
        if not text1 or not text2:
            return 0.0

        # Counter 在函数内部导入，避免模块级额外依赖。
        from collections import Counter

        # Character frequency (TF)
        # 统计两个文本的字符频次。
        tf1 = Counter(text1)
        tf2 = Counter(text2)

        # All unique characters (vocabulary)
        # 构建字符词表。
        vocab = set(tf1.keys()) | set(tf2.keys())

        # Simple IDF: log(2 / df) where df is document frequency
        # For two documents, IDF is log(2/1)=0.693 if char appears in one doc,
        # or log(2/2)=0 if appears in both (we use log(2/1) for simplicity)
        # 这里用 1.0/1.5 的简化权重替代标准 IDF。
        idf = {char: (1.0 if char in tf1 and char in tf2 else 1.5) for char in vocab}

        # TF-IDF vectors
        # 为两个文本构造向量。
        vec1 = {char: tf1.get(char, 0) * idf[char] for char in vocab}
        vec2 = {char: tf2.get(char, 0) * idf[char] for char in vocab}

        # Cosine similarity
        # 点积衡量共同方向。
        dot_product = sum(vec1[char] * vec2[char] for char in vocab)

        # 分别计算向量长度。
        norm1 = math.sqrt(sum(v * v for v in vec1.values()))
        norm2 = math.sqrt(sum(v * v for v in vec2.values()))

        # 零向量无法计算余弦相似度，返回 0。
        if norm1 == 0 or norm2 == 0:
            return 0.0

        # 返回余弦相似度。
        return dot_product / (norm1 * norm2)

    # 运行时实际生效的高度相似判断函数。
    # 它被 MMR 去重的预填阶段和选择阶段调用。
    @staticmethod
    def _is_text_highly_similar_optimized(
        candidate_idx: int,
        candidate_text: str,
        selected_global: list[int],
        similarity_matrix,
        flat: list,
        threshold: float = 0.92,
    ) -> bool:
        """
        Multi-algorithm text similarity check with embedding pre-filtering.

        Strategy:
        1. Only compare with the single highest embedding similarity item (not all 25)
        2. Only perform text comparison if embedding similarity > 0.60
        3. Use weighted combination of three algorithms:
           - Dice (40%): Fastest, character-level set similarity
           - TF-IDF (35%): Considers character frequency weighting
           - 2-gram (25%): Considers local character order

        Combined formula:
            combined_score = 0.40 * dice + 0.35 * tfidf + 0.25 * bigram

        This reduces comparisons from O(N) to O(1) per candidate, with embedding pre-filtering.
        Expected speedup: 100-200x compared to LCS approach.

        Args:
            candidate_idx: Index of candidate memory in flat list
            candidate_text: Text content of candidate memory
            selected_global: List of already selected memory indices
            similarity_matrix: Precomputed embedding similarity matrix
            flat: Flat list of all memories
            threshold: Combined similarity threshold (default 0.75)

        Returns:
            True if candidate is highly similar to any selected memory
        """
        # 没有已选结果时，候选不可能与已选结果重复。
        if not selected_global:
            return False

        # Find the already-selected memory with highest embedding similarity
        # 只找 embedding 最接近的已选项，降低文本比较成本。
        max_sim_idx = max(selected_global, key=lambda j: similarity_matrix[candidate_idx][j])

        # 取出候选与该已选项的 embedding 相似度。
        max_sim = similarity_matrix[candidate_idx][max_sim_idx]

        # If highest embedding similarity < 0.60, skip text comparison entirely
        # 实际代码使用 0.9 作为 embedding 预过滤门槛。
        # 只有 embedding 已经非常接近，才进一步做文本相似度组合判断。
        if max_sim <= 0.9:
            return False

        # Get text of most similar memory
        # 找到最相似已选记忆的文本内容。
        most_similar_mem = flat[max_sim_idx][2]
        most_similar_text = most_similar_mem.get("memory", "").strip()

        # Calculate three similarity scores
        # 分别计算字符集合、字符频率、局部 2-gram 三个维度的相似度。
        dice_sim = SearchHandler._dice_similarity(candidate_text, most_similar_text)
        tfidf_sim = SearchHandler._tfidf_similarity(candidate_text, most_similar_text)
        bigram_sim = SearchHandler._bigram_similarity(candidate_text, most_similar_text)

        # Weighted combination: Dice (40%) + TF-IDF (35%) + 2-gram (25%)
        # Dice has highest weight (fastest and most reliable)
        # TF-IDF considers frequency (handles repeated characters well)
        # 2-gram considers order (catches local pattern similarity)
        # 加权合并为最终文本相似度。
        combined_score = 0.40 * dice_sim + 0.35 * tfidf_sim + 0.25 * bigram_sim

        # 达到阈值时，认为当前候选与已选集合高度相似。
        return combined_score >= threshold

    # 解析本次搜索可读取的 cube IDs。
    # 搜索和新增类似，都支持多 cube；区别是这里使用 readable_cube_ids。
    def _resolve_cube_ids(self, search_req: APISearchRequest) -> list[str]:
        """
        Normalize target cube ids from search_req.
        Priority:
        1) readable_cube_ids (deprecated mem_cube_id is converted to this in model validator)
        2) fallback to user_id
        """
        # 如果请求显式提供 readable_cube_ids，则优先使用它。
        if search_req.readable_cube_ids:
            # dict.fromkeys 去重并保留原始顺序。
            return list(dict.fromkeys(search_req.readable_cube_ids))

        # 如果没有指定可读 cube，则默认搜索用户自己的 cube。
        return [search_req.user_id]

    # 根据目标 cube 数量构建搜索视图。
    # 单 cube 返回 SingleCubeView，多 cube 返回 CompositeCubeView。
    def _build_cube_view(self, search_req: APISearchRequest, searcher=None) -> MemCubeView:
        # 解析搜索目标 cube。
        cube_ids = self._resolve_cube_ids(search_req)

        # 如果调用方传入自定义 searcher，则使用它；否则使用 handler 默认 searcher。
        # 这让测试或特殊搜索路径可以临时替换底层 searcher。
        searcher_to_use = searcher if searcher is not None else self.searcher

        # 单 cube 搜索路径。
        if len(cube_ids) == 1:
            # 取出唯一 cube_id。
            cube_id = cube_ids[0]

            # 构建单 cube 视图。
            # 搜索逻辑最终会委托给 SingleCubeView.search_memories。
            return SingleCubeView(
                cube_id=cube_id,
                naive_mem_cube=self.naive_mem_cube,
                mem_reader=self.mem_reader,
                mem_scheduler=self.mem_scheduler,
                logger=self.logger,
                searcher=searcher_to_use,
                deepsearch_agent=self.deepsearch_agent,
            )

        # 多 cube 搜索路径。
        else:
            # 为每个 cube_id 创建一个 SingleCubeView。
            # 这些视图共享相同底层依赖，但绑定不同 cube_id。
            single_views = [
                SingleCubeView(
                    cube_id=cube_id,
                    naive_mem_cube=self.naive_mem_cube,
                    mem_reader=self.mem_reader,
                    mem_scheduler=self.mem_scheduler,
                    logger=self.logger,
                    searcher=searcher_to_use,
                    deepsearch_agent=self.deepsearch_agent,
                )
                # 遍历所有可读 cube。
                for cube_id in cube_ids
            ]

            # 用 CompositeCubeView 统一包装多个 SingleCubeView。
            # 上层只需要调用一次 search_memories，组合视图负责分发和聚合。
            return CompositeCubeView(cube_views=single_views, logger=self.logger)
