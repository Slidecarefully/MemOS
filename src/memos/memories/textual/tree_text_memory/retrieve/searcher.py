# 导入 copy，用于在 keyword/fulltext 路径中复制节点字典，避免修改 graph_store 返回的原始结构。
import copy
# 导入 importlib，用于按需动态加载 jieba.analyse，避免中文关键词抽取依赖在模块加载阶段强制初始化。
import importlib
# 导入 re，用于英文关键词切分、正则识别 token，以及 keyword term 规范化。
import re
# 导入 traceback，用于 fine_old 分支中记录节点解析异常的完整堆栈。
import traceback

# 导入 as_completed，用于 rawfile 去重时并发读取 edge，并按完成顺序收集结果。
from concurrent.futures import as_completed

# 导入上下文感知线程池，保证并发召回任务能继承请求上下文和日志 trace。
from memos.context.context import ContextThreadPoolExecutor
# 导入 OllamaEmbedder 类型，用于给 Searcher 标注 embedding 组件依赖。
from memos.embedders.factory import OllamaEmbedder
# 导入 Neo4jGraphDB 类型，Searcher 通过它访问图数据库中的 memory 节点和边。
from memos.graph_dbs.factory import Neo4jGraphDB
# 导入不同 LLM 客户端类型，dispatcher_llm 可来自 OpenAI、Ollama 或 Azure。
from memos.llms.factory import AzureLLM, OllamaLLM, OpenAILLM
# 导入日志构造函数，当前模块会大量记录搜索路径、召回数量、异常和最终结果。
from memos.log import get_logger
# 导入文本记忆数据结构；搜索结果最终会被包装成带 relativity 元数据的 TextualMemoryItem。
from memos.memories.textual.item import SearchedTreeNodeTextualMemoryMetadata, TextualMemoryItem
# 导入 BM25 检索器类型，GraphMemoryRetriever 可借助它做关键词/混合召回。
from memos.memories.textual.tree_text_memory.retrieve.bm25_util import EnhancedBM25
# 导入检索工具函数，包括分词、停用词、语言检测、相似度矩阵和 JSON 解析等。
from memos.memories.textual.tree_text_memory.retrieve.retrieve_utils import (
    # FastTokenizer 用于英文或混合文本分词，支撑 keyword 和 plugin 简化检索。
    FastTokenizer,
    # StopwordManager 负责判断搜索关键词是否属于停用词。
    StopwordManager,
    # cosine_similarity_matrix 用于计算候选文档之间的 embedding 相似度矩阵。
    cosine_similarity_matrix,
    # detect_lang 用于判断 query 是中文还是英文，从而选择关键词抽取和 COT prompt。
    detect_lang,
    # find_best_unrelated_subgroup 用于从候选中挑出彼此不太相似的一组结果。
    find_best_unrelated_subgroup,
    # parse_json_result 用于把 LLM 生成的 JSON 文本解析成 Python dict。
    parse_json_result,
)
# 导入 reranker 基类，Searcher 用它对各路径召回结果做二次排序。
from memos.reranker.base import BaseReranker
# 导入搜索 query 拆解用的 prompt 模板，fast/fine 和中英文会选择不同模板。
from memos.templates.mem_search_prompts import (
    # COT_PROMPT 是英文 fine 模式 query 拆解模板。
    COT_PROMPT,
    # COT_PROMPT_ZH 是中文 fine 模式 query 拆解模板。
    COT_PROMPT_ZH,
    # SIMPLE_COT_PROMPT 是英文 fast 模式简化 query 拆解模板。
    SIMPLE_COT_PROMPT,
    # SIMPLE_COT_PROMPT_ZH 是中文 fast 模式简化 query 拆解模板。
    SIMPLE_COT_PROMPT_ZH,
)
# 导入 timed 装饰器，用于记录搜索关键阶段的耗时。
from memos.utils import timed

# 导入 MemoryReasoner，负责搜索链路中的推理组件初始化。
from .reasoner import MemoryReasoner
# 导入 GraphMemoryRetriever，它是真正访问 graph_store 并召回节点的底层组件。
from .recall import GraphMemoryRetriever
# 导入 TaskGoalParser，用于把用户 query 解析成检索目标、重写 query 和辅助 memories。
from .task_goal_parser import TaskGoalParser


# 创建模块级 logger，使日志能标记到当前 searcher 模块。
logger = get_logger(__name__)
# 设置关键词抽取数量上限，fulltext 路径最多使用 3 个加权关键词。
KEYWORD_EXTRACT_TOP_K = 3
# 定义中文 jieba 关键词抽取允许的词性，偏向名词、动词、时间、英文和数字等可检索实体。
KEYWORD_ALLOW_POS = ("n", "nr", "nrt", "ns", "nt", "nz", "vn", "v", "t", "eng", "m")
# 建立 COT query 拆解 prompt 映射，根据搜索模式和语言选择对应模板。
COT_DICT = {
    # fine 模式使用更完整的中英文 COT prompt，倾向于精细拆解复杂问题。
    "fine": {"en": COT_PROMPT, "zh": COT_PROMPT_ZH},
    # fast 模式使用简化 prompt，减少 LLM 负担以提升检索速度。
    "fast": {"en": SIMPLE_COT_PROMPT, "zh": SIMPLE_COT_PROMPT_ZH},
}


# 定义 Searcher：它是文本记忆搜索核心，负责 query 解析、多路径召回、rerank、去重、结果裁剪和使用记录。
class Searcher:
    # 初始化 Searcher 的依赖组件和检索策略开关。
    def __init__(
        self,
        dispatcher_llm: OpenAILLM | OllamaLLM | AzureLLM,
        graph_store: Neo4jGraphDB,
        embedder: OllamaEmbedder,
        reranker: BaseReranker,
        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
        bm25_retriever: EnhancedBM25 | None = None,
        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
        internet_retriever: None = None,
        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
        search_strategy: dict | None = None,
        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
        manual_close_internet: bool = True,
        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
        tokenizer: FastTokenizer | None = None,
        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
        include_embedding: bool = False,
    ):
        # 保存图数据库句柄，后续所有 memory 节点/边召回都通过它完成。
        self.graph_store = graph_store
        # 保存 embedding 组件，用于 query、COT query、候选文档等向量化。
        self.embedder = embedder
        # 保存 LLM，用于任务目标解析和 COT 子问题生成。
        self.llm = dispatcher_llm

        # 初始化任务目标解析器，把自然语言 query 变成结构化检索目标。
        self.task_goal_parser = TaskGoalParser(dispatcher_llm)
        # 初始化图记忆召回器，封装 embedding/BM25/图数据库混合检索能力。
        self.graph_retriever = GraphMemoryRetriever(
            # 把图数据库、embedding、BM25 和是否返回 embedding 的配置传给召回器。
            graph_store, embedder, bm25_retriever, include_embedding=include_embedding
        )
        # 保存 reranker，用于在各召回路径之后做相关性重排。
        self.reranker = reranker
        # 初始化记忆推理器，保留给需要 LLM 推理筛选的搜索链路使用。
        self.reasoner = MemoryReasoner(dispatcher_llm)

        # Create internet retriever from config if provided
        # 保存外部互联网检索器；没有配置时 Path C 会跳过。
        self.internet_retriever = internet_retriever
        # 读取是否启用向量版 COT 检索：启用后会把拆解出的子问题也做 embedding。
        self.vec_cot = search_strategy.get("cot", False) if search_strategy else False
        # 读取 fast_graph 开关，传给 GraphMemoryRetriever 选择更快的图召回策略。
        self.use_fast_graph = search_strategy.get("fast_graph", False) if search_strategy else False
        # 读取 fulltext 开关，决定是否额外启动 keyword/fulltext 检索路径。
        self.use_fulltext = search_strategy.get("fulltext", False) if search_strategy else False
        # 保存互联网检索手动关闭策略；为 True 时只有 parsed_goal 要求联网才会检索互联网。
        self.manual_close_internet = manual_close_internet
        # 保存可选 tokenizer；没有时部分路径会临时创建 FastTokenizer 或用 split 兜底。
        self.tokenizer = tokenizer
        # 创建 usage history 后台线程池，避免搜索主流程被数据库写 usage 操作阻塞。
        self._usage_executor = ContextThreadPoolExecutor(max_workers=4, thread_name_prefix="usage")

    # 根据 rerank 开关决定是否调用 reranker；关闭时保留原始召回顺序并给默认分数。
    def _maybe_rerank(
        self,
        enabled: bool,
        *,
        query: str,
        graph_results: list[TextualMemoryItem],
        top_k: int,
        # 透传 parsed_goal、search_filter、query_embedding 等路径相关参数。
        **kwargs,
    ) -> list[tuple[TextualMemoryItem, float]]:
        # 如果 rerank 没开启或没有 reranker 实例，就走轻量兜底路径。
        if not enabled or self.reranker is None:
            # rerank 关闭或 reranker 不存在时，直接截取前 top_k 个结果，并给默认分数 0。
            return [(item, 0.0) for item in graph_results[:top_k]]
        # rerank 开启时，把候选结果交给 reranker 重新评分排序。
        return self.reranker.rerank(
            # 把原始或重写后的 query 传给 reranker，作为相关性判断基准。
            query=query,
            # 把当前召回路径得到的候选 memory 列表传入 reranker。
            graph_results=graph_results,
            # 限制 reranker 最终返回的候选数量。
            top_k=top_k,
            # 透传 parsed_goal、search_filter、query_embedding 等路径相关参数。
            **kwargs,
        )

    # 声明为静态方法，表示该逻辑不依赖 Searcher 实例状态。
    @staticmethod
    # 将 embedding 批结果转换为 reranker 期望的单个 query embedding。
    def _query_embedding_for_rerank(enabled: bool, query_embedding):
        # 如果调用方没有启用对应能力，就返回空/默认值，避免做无意义计算。
        if not enabled:
            # 返回 None，表示下游不应使用该可选值。
            return None
        # reranker 通常只需要原始 query 的 embedding，因此取批量 embedding 的第一个元素。
        return query_embedding[0]

    # 用 timed 装饰当前阶段，方便日志或监控统计耗时。
    @timed
    # 执行搜索的前半段：解析 query，并行跑多条召回路径，返回带分数的候选结果。
    def retrieve(
        self,
        query: str,
        top_k: int,
        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
        info=None,
        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
        mode="fast",
        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
        memory_type="All",
        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
        search_filter: dict | None = None,
        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
        search_priority: dict | None = None,
        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
        user_name: str | None = None,
        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
        search_tool_memory: bool = False,
        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
        tool_mem_top_k: int = 6,
        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
        include_skill_memory: bool = False,
        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
        skill_mem_top_k: int = 3,
        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
        include_preference_memory: bool = False,
        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
        pref_mem_top_k: int = 6,
        **kwargs,
    ) -> list[tuple[TextualMemoryItem, float]]:
        # 记录关键运行状态，方便追踪搜索路径、候选数量或分支选择。
        logger.info(
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            f"[RECALL] Start query='{query}', top_k={top_k}, mode={mode}, memory_type={memory_type}, user_name={user_name}"
        )
        # 从额外参数中读取 rerank 开关，默认启用重排。
        rerank = bool(kwargs.get("rerank", True))
        # 先解析任务目标，同时可能得到 query embedding、上下文和重写后的 query。
        parsed_goal, query_embedding, _context, query = self._parse_task(
            query,
            info,
            mode,
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            search_filter=search_filter,
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            search_priority=search_priority,
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            user_name=user_name,
            **kwargs,
        )
        # 基于解析结果并行跑多条检索路径，得到原始候选。
        results = self._retrieve_paths(
            query,
            parsed_goal,
            query_embedding,
            info,
            top_k,
            mode,
            memory_type,
            search_filter,
            search_priority,
            user_name,
            search_tool_memory,
            tool_mem_top_k,
            include_skill_memory,
            skill_mem_top_k,
            include_preference_memory,
            pref_mem_top_k,
            rerank,
        )
        # 返回当前阶段处理后的结果，交给上层继续后处理。
        return results

    # 执行搜索的后半段：去重、分类裁剪、写入 usage history，得到最终结果列表。
    def post_retrieve(
        self,
        retrieved_results: list[tuple[TextualMemoryItem, float]],
        top_k: int,
        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
        user_name: str | None = None,
        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
        info=None,
        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
        search_tool_memory: bool = False,
        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
        tool_mem_top_k: int = 6,
        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
        include_skill_memory: bool = False,
        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
        skill_mem_top_k: int = 3,
        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
        include_preference_memory: bool = False,
        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
        pref_mem_top_k: int = 6,
        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
        dedup: str | None = None,
        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
        plugin=False,
    ):
        # 如果调用方明确关闭去重，就保留全部召回结果。
        if dedup == "no":
            # 不做文本去重，直接沿用原候选。
            deduped = retrieved_results
        # 默认路径会做文本级去重或进入替代分支。
        else:
            # 按 memory 文本去重，避免多条路径召回同一内容。
            deduped = self._deduplicate_results(retrieved_results)
        # 按记忆类型和分数排序截断，生成最终返回 item。
        final_results = self._sort_and_trim(
            deduped,
            top_k,
            plugin,
            search_tool_memory,
            tool_mem_top_k,
            include_skill_memory,
            skill_mem_top_k,
            include_preference_memory,
            pref_mem_top_k,
        )
        # 异步记录这些 memory 被使用过，便于后续统计或排序策略使用。
        self._update_usage_history(final_results, info, user_name)
        # 返回最终整理后的搜索结果列表。
        return final_results

    # 用 timed 装饰当前阶段，方便日志或监控统计耗时。
    @timed
    # 对外主搜索入口：整合 info 校验、plugin 分支、召回、后处理和日志输出。
    def search(
        self,
        query: str,
        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
        top_k: int = 10,
        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
        info=None,
        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
        mode="fast",
        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
        memory_type="All",
        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
        search_filter: dict | None = None,
        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
        search_priority: dict | None = None,
        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
        user_name: str | None = None,
        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
        search_tool_memory: bool = False,
        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
        tool_mem_top_k: int = 6,
        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
        include_skill_memory: bool = False,
        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
        skill_mem_top_k: int = 3,
        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
        include_preference_memory: bool = False,
        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
        pref_mem_top_k: int = 6,
        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
        dedup: str | None = None,
        **kwargs,
    ) -> list[TextualMemoryItem]:
        """
        Search for memories based on a query.
        User query -> TaskGoalParser -> GraphMemoryRetriever ->
        MemoryReranker -> MemoryReasoner -> Final output
        Args:
            query (str): The query to search for.
            top_k (int): The number of top results to return.
            info (dict): Leave a record of memory consumption.
            mode (str, optional): The mode of the search.
            - 'fast': Uses a faster search process, sacrificing some precision for speed.
            - 'fine': Uses a more detailed search process, invoking large models for higher precision, but slower performance.
            memory_type (str): Type restriction for search.
            ['All', 'WorkingMemory', 'LongTermMemory', 'UserMemory']
            search_filter (dict, optional): Optional metadata filters for search results.
            search_priority (dict, optional): Optional metadata priority for search results.
        Returns:
            list[TextualMemoryItem]: List of matching memories.
        """
        # 如果没有传 info，搜索仍继续，但会用空 user/session 信息兜底。
        if not info:
            # 记录非致命异常或配置问题，不中断主搜索流程。
            logger.warning(
                "Please input 'info' when use tree.search so that "
                "the database would store the consume history."
            )
            # 构造最小 info，避免后续读取 user_id/session_id 时出错。
            info = {"user_id": "", "session_id": ""}
        # 非 plugin 模式走完整 Searcher 检索链路。
        else:
            # 调试日志记录 info 内容，帮助排查 session/user 过滤问题。
            logger.debug(f"[SEARCH] Received info dict: {info}")

        # plugin 模式走简化召回，不使用完整任务解析和多路径检索。
        if kwargs.get("plugin", False):
            # 调用相关组件完成当前子步骤，并把结果交给后续逻辑。
            logger.info(f"[SEARCH] Retrieve from plugin: {query}")
            # 插件来源使用 query/关键词/embedding 混合召回的轻量路径。
            retrieved_results = self._retrieve_simple(
                # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
                query=query, top_k=top_k, search_filter=search_filter, user_name=user_name
            )
        # 非 plugin 模式走完整 Searcher 检索链路。
        else:
            # 先执行 query 解析和并行多路径召回。
            retrieved_results = self.retrieve(
                # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
                query=query,
                # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
                top_k=top_k,
                # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
                info=info,
                # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
                mode=mode,
                # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
                memory_type=memory_type,
                # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
                search_filter=search_filter,
                # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
                search_priority=search_priority,
                # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
                user_name=user_name,
                # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
                search_tool_memory=search_tool_memory,
                # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
                tool_mem_top_k=tool_mem_top_k,
                # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
                include_skill_memory=include_skill_memory,
                # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
                skill_mem_top_k=skill_mem_top_k,
                # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
                include_preference_memory=include_preference_memory,
                # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
                pref_mem_top_k=pref_mem_top_k,
                **kwargs,
            )

        # 读取 full_recall 开关；开启时返回原始候选，不做最终裁剪。
        full_recall = kwargs.get("full_recall", False)
        # 如果调用方要完整召回结果，就跳过 post_retrieve。
        if full_recall:
            # 直接返回带分数的原始召回结果。
            return retrieved_results

        # 对召回结果做去重、排序、裁剪和 usage 更新。
        final_results = self.post_retrieve(
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            retrieved_results=retrieved_results,
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            top_k=top_k,
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            user_name=user_name,
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            info=None,
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            plugin=kwargs.get("plugin", False),
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            search_tool_memory=search_tool_memory,
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            tool_mem_top_k=tool_mem_top_k,
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            include_skill_memory=include_skill_memory,
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            skill_mem_top_k=skill_mem_top_k,
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            include_preference_memory=include_preference_memory,
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            pref_mem_top_k=pref_mem_top_k,
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            dedup=dedup,
        )

        # 调用相关组件完成当前子步骤，并把结果交给后续逻辑。
        logger.info(f"[SEARCH] Done. Total {len(final_results)} results.")
        # 准备拼接结果摘要日志。
        res_results = ""
        # 遍历最终结果，用于构造调试日志中的结果摘要。
        for _num_i, result in enumerate(final_results):
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            res_results += "\n" + (
                # 日志中按 id、memory_type、memory 内容拼接，方便观察最终命中的节点。
                result.id + "|" + result.metadata.memory_type + "|" + result.memory
            )
        # 调用相关组件完成当前子步骤，并把结果交给后续逻辑。
        logger.info(f"[SEARCH] Results. {res_results}")
        # 返回最终整理后的搜索结果列表。
        return final_results

    # 用 timed 装饰当前阶段，方便日志或监控统计耗时。
    @timed
    # 解析用户 query：可选先做初始 embedding 上下文召回，再用 LLM 解析任务目标和重写 query。
    def _parse_task(
        self,
        query,
        info,
        mode,
        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
        top_k=5,
        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
        search_filter: dict | None = None,
        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
        search_priority: dict | None = None,
        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
        user_name: str | None = None,
        **kwargs,
    ):
        """Parse user query, do embedding search and create context"""
        # 初始化解析上下文；fine_old 会先召回相关 memory 填充它。
        context = []
        # 初始化 query_embedding；不是所有模式都会立刻生成 embedding。
        query_embedding = None

        # fine mode will trigger initial embedding search
        # 旧版 fine 模式先做一次 embedding 搜索，把相关记忆作为 LLM 解析上下文。
        if mode == "fine_old":
            # 调用相关组件完成当前子步骤，并把结果交给后续逻辑。
            logger.info("[SEARCH] Fine mode: embedding search")
            # 把原始 query 转成单个 embedding，用于初始向量召回。
            query_embedding = self.embedder.embed([query])[0]

            # retrieve related nodes by embedding
            # 根据 query embedding 召回初始相关节点，用作任务解析上下文。
            related_nodes = [
                # 通过搜索命中的节点 id 再读取完整节点内容。
                self.graph_store.get_node(n["id"], user_name=user_name)
                # 遍历 embedding 搜索命中的节点摘要。
                for n in self.graph_store.search_by_embedding(
                    # 传入 query 的向量表示作为向量搜索条件。
                    query_embedding,
                    # 限制初始上下文召回数量。
                    top_k=top_k,
                    # 只召回已激活 memory，避免已删除或未启用节点进入上下文。
                    status="activated",
                    # 把优先级过滤传给 embedding 搜索。
                    search_filter=search_priority,
                    # 传入普通 metadata filter，限定检索范围。
                    filter=search_filter,
                    # 限定用户/cube 命名空间，避免跨用户读取记忆。
                    user_name=user_name,
                )
            ]
            # 收集可作为 LLM 上下文的 memory 文本。
            memories = []
            # 遍历初始召回节点，提取其中的 memory 字段。
            for node in related_nodes:
                # 开始保护性执行，避免某个子步骤异常拖垮整条搜索链路。
                try:
                    # 兼容 dict 节点和对象节点两种结构读取 memory 内容。
                    m = (
                        # 调用相关组件完成当前子步骤，并把结果交给后续逻辑。
                        node.get("memory")
                        # 根据当前配置、输入参数或中间结果选择是否进入该分支。
                        if isinstance(node, dict)
                        # 调用相关组件完成当前子步骤，并把结果交给后续逻辑。
                        else (getattr(node, "memory", None))
                    )
                    # 只有非空字符串才进入上下文。
                    if isinstance(m, str) and m:
                        # 把有效 memory 文本加入上下文候选。
                        memories.append(m)
                # 兜底捕获异常，当前实现选择记录日志并降级处理。
                except Exception:
                    # 调用相关组件完成当前子步骤，并把结果交给后续逻辑。
                    logger.error(f"[SEARCH] Error during search: {traceback.format_exc()}")
                    # 跳过当前候选，继续处理后续数据。
                    continue
            # 对上下文 memory 去重并保留原顺序。
            context = list(dict.fromkeys(memories))

            # optional: supplement context with internet knowledge
            """if self.internet_retriever:
                extra = self.internet_retriever.retrieve_from_internet(query=query, top_k=3)
                context.extend(item.memory.partition("\nContent: ")[-1] for item in extra)
            """

        # parse goal using LLM
        # 调用 LLM/规则解析器，把 query 转为结构化检索目标。
        parsed_goal = self.task_goal_parser.parse(
            # 把原始用户查询作为任务描述传入解析器。
            task_description=query,
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            context="\n".join(context),
            # 把历史对话传入解析器，支持上下文相关查询。
            conversation=info.get("chat_history", []),
            # 传入搜索模式，让解析器选择 fast/fine 行为。
            mode=mode,
            # 把 fast_graph 策略传给目标解析器，影响后续图检索目标生成。
            use_fast_graph=self.use_fast_graph,
            **kwargs,
        )

        # 如果解析器重写了 query，就用重写版本进行后续召回。
        query = parsed_goal.rephrased_query or query
        # if goal has extra memories, embed them too
        # 如果解析器额外抽取了辅助 memory 文本，就一起生成 embedding。
        if parsed_goal.memories:
            # 把重写 query 和辅助 memories 去重后组成 embedding 输入。
            embed_texts = list(dict.fromkeys([query, *parsed_goal.memories]))
            # 批量生成 query/辅助 memory embeddings，供多路径召回使用。
            query_embedding = self.embedder.embed(embed_texts)
        # 返回解析目标、embedding、上下文和最终 query，供 retrieve 使用。
        return parsed_goal, query_embedding, context, query

    # 用 timed 装饰当前阶段，方便日志或监控统计耗时。
    @timed
    # 并行运行多条检索路径：WorkingMemory、LongTerm/User、Internet、Keyword、Tool、Skill、Preference。
    def _retrieve_paths(
        self,
        query,
        parsed_goal,
        query_embedding,
        info,
        top_k,
        mode,
        memory_type,
        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
        search_filter: dict | None = None,
        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
        search_priority: dict | None = None,
        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
        user_name: str | None = None,
        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
        search_tool_memory: bool = False,
        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
        tool_mem_top_k: int = 6,
        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
        include_skill_memory: bool = False,
        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
        skill_mem_top_k: int = 3,
        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
        include_preference_memory: bool = False,
        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
        pref_mem_top_k: int = 6,
        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
        rerank: bool = True,
    ):
        """Run A/B/C/D/E/F retrieval paths in parallel"""
        # 收集并发提交的检索任务 future。
        tasks = []
        # 构造基于 user_id/session_id 的过滤条件，用于限制当前会话相关记忆。
        id_filter = {
            # 从 info 中提取 user_id 过滤条件。
            "user_id": info.get("user_id", None),
            # 从 info 中提取 session_id 过滤条件。
            "session_id": info.get("session_id", None),
        }
        # 去掉值为 None 的过滤项，避免传入无效 filter。
        id_filter = {k: v for k, v in id_filter.items() if v is not None}

        # 使用上下文线程池并发跑基础检索路径，降低总体搜索延迟。
        with ContextThreadPoolExecutor(max_workers=5) as executor:
            # 向并发任务列表追加一个检索路径。
            tasks.append(
                # 把某个检索路径提交到线程池执行。
                executor.submit(
                    # 提交 Path A：WorkingMemory 召回。
                    self._retrieve_from_working_memory,
                    query,
                    parsed_goal,
                    query_embedding,
                    top_k,
                    memory_type,
                    search_filter,
                    search_priority,
                    user_name,
                    id_filter,
                    # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
                    rerank=rerank,
                )
            )
            # 向并发任务列表追加一个检索路径。
            tasks.append(
                # 把某个检索路径提交到线程池执行。
                executor.submit(
                    # 提交 Path B：LongTerm/User/RawFile 召回。
                    self._retrieve_from_long_term_and_user,
                    query,
                    parsed_goal,
                    query_embedding,
                    top_k,
                    memory_type,
                    search_filter,
                    search_priority,
                    user_name,
                    id_filter,
                    # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
                    mode=mode,
                    # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
                    rerank=rerank,
                )
            )
            # 向并发任务列表追加一个检索路径。
            tasks.append(
                # 把某个检索路径提交到线程池执行。
                executor.submit(
                    # 提交 Path C：互联网召回。
                    self._retrieve_from_internet,
                    query,
                    parsed_goal,
                    query_embedding,
                    top_k,
                    info,
                    mode,
                    memory_type,
                    user_name,
                    # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
                    rerank=rerank,
                )
            )
            # 如果启用 fulltext 策略，额外启动 keyword 检索路径。
            if self.use_fulltext:
                # 向并发任务列表追加一个检索路径。
                tasks.append(
                    # 把某个检索路径提交到线程池执行。
                    executor.submit(
                        # 提交 keyword/fulltext 检索路径。
                        self._retrieve_from_keyword,
                        query,
                        parsed_goal,
                        query_embedding,
                        top_k,
                        memory_type,
                        search_filter,
                        search_priority,
                        user_name,
                        id_filter,
                        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
                        rerank=rerank,
                    )
                )
            # 如果请求包含工具记忆，则追加 ToolMemory 检索。
            if search_tool_memory:
                # 向并发任务列表追加一个检索路径。
                tasks.append(
                    # 把某个检索路径提交到线程池执行。
                    executor.submit(
                        # 提交 ToolSchema/ToolTrajectory 检索路径。
                        self._retrieve_from_tool_memory,
                        query,
                        parsed_goal,
                        query_embedding,
                        tool_mem_top_k,
                        memory_type,
                        search_filter,
                        search_priority,
                        user_name,
                        id_filter,
                        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
                        mode=mode,
                        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
                        rerank=rerank,
                    )
                )
            # 如果请求包含技能记忆，则追加 SkillMemory 检索。
            if include_skill_memory:
                # 向并发任务列表追加一个检索路径。
                tasks.append(
                    # 把某个检索路径提交到线程池执行。
                    executor.submit(
                        # 提交技能记忆检索路径。
                        self._retrieve_from_skill_memory,
                        query,
                        parsed_goal,
                        query_embedding,
                        skill_mem_top_k,
                        memory_type,
                        search_filter,
                        search_priority,
                        user_name,
                        id_filter,
                        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
                        mode=mode,
                        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
                        rerank=rerank,
                    )
                )
            # 如果请求包含偏好记忆，则追加 PreferenceMemory 检索。
            if include_preference_memory:
                # 向并发任务列表追加一个检索路径。
                tasks.append(
                    # 把某个检索路径提交到线程池执行。
                    executor.submit(
                        # 提交偏好记忆检索路径。
                        self._retrieve_from_preference_memory,
                        query,
                        parsed_goal,
                        query_embedding,
                        pref_mem_top_k,
                        memory_type,
                        search_filter,
                        search_priority,
                        user_name,
                        id_filter,
                        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
                        mode=mode,
                        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
                        rerank=rerank,
                    )
                )
            # 收集所有检索路径返回的候选。
            results = []
            # 遍历每个 future，等待并合并结果。
            for t in tasks:
                # 取出子路径结果并追加到总候选列表；若子路径异常会在这里抛出。
                results.extend(t.result())

        # 调用相关组件完成当前子步骤，并把结果交给后续逻辑。
        logger.info(f"[SEARCH] Total raw results: {len(results)}")
        # 返回当前阶段处理后的结果，交给上层继续后处理。
        return results

    # --- Path A
    # 用 timed 装饰当前阶段，方便日志或监控统计耗时。
    @timed
    # Path A：从 WorkingMemory 中召回短期/工作记忆并可选 rerank。
    def _retrieve_from_working_memory(
        self,
        query,
        parsed_goal,
        query_embedding,
        top_k,
        memory_type,
        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
        search_filter: dict | None = None,
        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
        search_priority: dict | None = None,
        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
        user_name: str | None = None,
        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
        id_filter: dict | None = None,
        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
        rerank: bool = True,
    ):
        """Retrieve and rerank from WorkingMemory"""
        # 如果用户限制的 memory_type 不包含 WorkingMemory，则当前路径直接跳过。
        if memory_type not in ["All", "WorkingMemory"]:
            # 调用相关组件完成当前子步骤，并把结果交给后续逻辑。
            logger.info(f"[PATH-A] '{query}'Skipped (memory_type does not match)")
            # 当前分支没有可用结果时返回空列表，让上层可以安全 extend。
            return []
        # 调用 GraphMemoryRetriever 从图数据库按指定 memory_scope 召回节点。
        items = self.graph_retriever.retrieve(
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            query=query,
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            parsed_goal=parsed_goal,
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            top_k=top_k,
            # 指定当前路径只召回 WorkingMemory 范围。
            memory_scope="WorkingMemory",
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            search_filter=search_filter,
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            search_priority=search_priority,
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            user_name=user_name,
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            id_filter=id_filter,
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            use_fast_graph=self.use_fast_graph,
        )
        # 当前路径召回完成后，按 rerank 开关决定是否重排并返回。
        return self._maybe_rerank(
            rerank,
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            query=query,
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            query_embedding=self._query_embedding_for_rerank(rerank, query_embedding),
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            graph_results=items,
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            top_k=top_k,
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            parsed_goal=parsed_goal,
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            search_filter=search_filter,
        )

    # 声明为静态方法，表示该逻辑不依赖 Searcher 实例状态。
    @staticmethod
    # 校验 fulltext 搜索必须提供 user_name，因为底层 PolarDB/图存储需要用户命名空间。
    def _require_keyword_user_name(user_name: str | None) -> str:
        # 把 user_name 规范化为空白去除后的字符串；非字符串视为空。
        normalized_user_name = user_name.strip() if isinstance(user_name, str) else ""
        # 没有有效 user_name 时不能执行用户命名空间内的 fulltext 检索。
        if not normalized_user_name:
            # 抛出明确错误，让调用方知道 fulltext 缺少必要 user_name。
            raise ValueError(
                "[PATH-KEYWORD] user_name is required for PolarDB fulltext keyword search"
            )
        # 返回规范化后的 user_name。
        return normalized_user_name

    # 声明为静态方法，表示该逻辑不依赖 Searcher 实例状态。
    @staticmethod
    # 判断关键词是否为空或停用词，避免把无意义词传入 fulltext。
    def _is_keyword_stopword(term: str) -> bool:
        # 去掉关键词两侧空白，统一停用词判断输入。
        normalized = term.strip()
        # 空关键词或停用词都不应该进入搜索。
        return not normalized or StopwordManager.is_search_stopword(normalized)

    # 声明为静态方法，表示该逻辑不依赖 Searcher 实例状态。
    @staticmethod
    # 规范化关键词：英文/数字 token 会转小写，中文或复杂符号保持原样。
    def _normalize_keyword_term(term: str) -> str:
        # 把 term 转成字符串并去掉空白。
        normalized = str(term).strip()
        # 如果 term 是英文/数字组合 token，就按大小写无关方式处理。
        if re.fullmatch(r"[A-Za-z][A-Za-z0-9]*(?:[._+\-/][A-Za-z0-9]+)*", normalized):
            # 英文 token 转小写，减少大小写导致的重复。
            return normalized.lower()
        # 中文或复杂 term 保持原样，避免破坏分词结果。
        return normalized

    # 声明为静态方法，表示该逻辑不依赖 Searcher 实例状态。
    @staticmethod
    # 根据 query 长度和语言决定抽取多少关键词，短 query 少抽，长 query 多抽。
    def _keyword_extract_top_k(query: str, language: str) -> int:
        # 清理 query 两侧空白，用长度判断关键词数量。
        cleaned_query = query.strip()
        # 空 query 不抽取关键词。
        if not cleaned_query:
            # 返回 0 表示 keyword 路径不应继续。
            return 0
        # 很短的 query 通常只有一个核心词。
        if len(cleaned_query) <= 12:
            # 短 query 只抽 1 个关键词，降低误召回。
            return 1
        # 非中文按英文 token 数量估计 query 复杂度。
        if language != "zh":
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            token_count = len(re.findall(r"\b[a-zA-Z0-9]+\b", cleaned_query))
            # 英文短句抽 2 个词，较长查询抽到上限。
            return 2 if token_count <= 8 else KEYWORD_EXTRACT_TOP_K
        # 中文中短 query 通常抽 2 个关键词即可。
        if len(cleaned_query) <= 120:
            # 中文中短 query 返回 2 个关键词。
            return 2
        # 长 query 使用关键词数量上限。
        return KEYWORD_EXTRACT_TOP_K

    # 声明为类方法，方便子类复用或覆盖格式化/排序逻辑。
    @classmethod
    # 对英文 token 按出现次数、长度、数字特征等打分排序，挑更适合全文检索的关键词。
    def _rank_english_keyword_terms(cls, terms: list[str]) -> list[str]:
        # 用字典统计每个英文 term 的首次位置和出现次数。
        term_stats: dict[str, dict[str, int | str]] = {}
        # 遍历 tokenizer 产出的英文候选词。
        for index, term in enumerate(terms):
            # 先规范化候选词。
            normalized_term = cls._normalize_keyword_term(term)
            # 停用词不参与关键词排名。
            if cls._is_keyword_stopword(normalized_term):
                # 跳过当前候选，继续处理后续数据。
                continue
            # 用小写 key 合并大小写不同的同一词。
            key = normalized_term.lower()
            # 第一次遇到该 term 时初始化统计信息。
            if key not in term_stats:
                # 记录标准 term、首次位置和计数。
                term_stats[key] = {"term": normalized_term, "index": index, "count": 0}
            # 增加该 term 的出现次数。
            term_stats[key]["count"] = int(term_stats[key]["count"]) + 1

        # 定义 score 方法，封装当前搜索链路中的一个独立步骤。
        def score(item: tuple[str, dict[str, int | str]]) -> tuple[float, int]:
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            _, data = item
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            term = str(data["term"])
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            count = int(data["count"])
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            term_score = count * 3.0 + min(len(term), 16) * 0.1
            # 根据当前配置、输入参数或中间结果选择是否进入该分支。
            if any(ch.isdigit() for ch in term):
                # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
                term_score += 1.0
            # 根据当前配置、输入参数或中间结果选择是否进入该分支。
            if len(term) <= 2:
                # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
                term_score -= 0.5
            # 返回当前步骤的处理结果，供上层搜索流程继续使用。
            return (-term_score, int(data["index"]))

        # 返回当前步骤的处理结果，供上层搜索流程继续使用。
        return [str(data["term"]) for _, data in sorted(term_stats.items(), key=score)]

    # 从 query 中抽取可用于 fulltext 的加权关键词，并完成停用词过滤和去重。
    def _extract_weighted_keyword_terms(self, query: str) -> list[str]:
        # 检测 query 语言，决定中文 jieba 还是英文 tokenizer。
        language = detect_lang(query)
        # 根据语言和长度决定抽取关键词数量。
        keyword_top_k = self._keyword_extract_top_k(query, language)
        # 没有可抽取关键词时直接返回空。
        if keyword_top_k <= 0:
            # 当前分支没有可用结果时返回空列表，让上层可以安全 extend。
            return []

        # 中文 query 使用 jieba.analyse 抽取关键词。
        if language == "zh":
            # 动态导入 jieba.analyse，只有中文关键词路径需要时才加载。
            jieba_analyse = importlib.import_module("jieba.analyse")

            # 调用 jieba TF-IDF 抽取中文关键词。
            weighted_terms = jieba_analyse.extract_tags(
                query,
                # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
                topK=keyword_top_k,
                # 只保留设定词性的关键词，减少虚词和无意义词。
                allowPOS=KEYWORD_ALLOW_POS,
            )
        # 当前面条件不满足时，执行默认或兜底逻辑。
        else:
            # 英文路径优先使用外部 tokenizer，没有则创建默认 FastTokenizer。
            tokenizer = self.tokenizer or FastTokenizer()
            # 对英文 token 排名，得到加权关键词候选。
            weighted_terms = self._rank_english_keyword_terms(tokenizer.tokenize_english(query))

        # 保存最终可用于全文检索的关键词。
        query_words: list[str] = []
        # 记录已选关键词，防止大小写或重复词重复进入查询。
        seen_words: set[str] = set()
        # 遍历加权候选关键词。
        for term in weighted_terms:
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            normalized_term = self._normalize_keyword_term(term)
            # 使用小写 key 做去重。
            dedupe_key = normalized_term.lower()
            # 过滤停用词和重复关键词。
            if self._is_keyword_stopword(normalized_term) or dedupe_key in seen_words:
                # 跳过当前候选，继续处理后续数据。
                continue
            # 记录当前关键词已被使用。
            seen_words.add(dedupe_key)
            # 加入最终关键词列表。
            query_words.append(normalized_term)
            # 达到目标关键词数量后停止。
            if len(query_words) >= keyword_top_k:
                # 达到目标条件后提前结束循环，避免不必要的额外处理。
                break
        # 返回最终关键词列表，供 fulltext 路径构造 tsquery。
        return query_words

    # 用 timed 装饰当前阶段，方便日志或监控统计耗时。
    @timed
    # Keyword/Fulltext 路径：把关键词转成 tsquery，查全文索引，再读取节点并 rerank。
    def _retrieve_from_keyword(
        self,
        query,
        parsed_goal,
        query_embedding,
        top_k,
        memory_type,
        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
        search_filter: dict | None = None,
        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
        search_priority: dict | None = None,
        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
        user_name: str | None = None,
        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
        id_filter: dict | None = None,
        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
        rerank: bool = True,
    ) -> list[tuple[TextualMemoryItem, float]]:
        """Keyword/fulltext path that directly calls graph DB fulltext search."""

        # keyword 路径只服务长期/用户记忆，其他类型直接跳过。
        if memory_type not in ["All", "LongTermMemory", "UserMemory"]:
            # 当前分支没有可用结果时返回空列表，让上层可以安全 extend。
            return []
        # 没有 query embedding 时无法执行依赖向量的 keyword/fulltext 后续读取重排。
        if not query_embedding:
            # 当前分支没有可用结果时返回空列表，让上层可以安全 extend。
            return []
        # fulltext 查询需要明确 user_name，因此先做校验和规范化。
        user_name = self._require_keyword_user_name(user_name)

        # 从用户 query 中抽取适合全文检索的关键词。
        query_words = self._extract_weighted_keyword_terms(query)
        # 如果没有有效关键词，fulltext 路径无法继续。
        if not query_words:
            # 当前分支没有可用结果时返回空列表，让上层可以安全 extend。
            return []
        # Quote weighted terms before `to_tsquery(...)` to avoid parsing operators from user input.
        # 把关键词包装成 tsquery 字面量并转义单引号，避免用户输入被当作查询操作符。
        tsquery_terms = ["'" + w.replace("'", "''") + "'" for w in query_words if w and w.strip()]
        # 转义后没有可查询 term 时直接返回。
        if not tsquery_terms:
            # 当前分支没有可用结果时返回空列表，让上层可以安全 extend。
            return []
        # 记录关键运行状态，方便追踪搜索路径、候选数量或分支选择。
        logger.info(
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            "[PATH-KEYWORD] weighted query_words=%s top_k=%s user_name=%s",
            query_words,
            top_k,
            user_name,
        )

        # 根据 memory_type 决定全文检索要覆盖哪些 scope。
        scopes = [memory_type] if memory_type != "All" else ["LongTermMemory", "UserMemory"]

        # 用节点 id 记录 fulltext 命中的最高分，处理多 scope 重复命中。
        id_to_score: dict[str, float] = {}
        # 逐个 memory scope 执行 fulltext 检索。
        for scope in scopes:
            # 开始保护性执行，避免某个子步骤异常拖垮整条搜索链路。
            try:
                # 调用 graph_store 的全文索引查询。
                hits = self.graph_store.search_by_fulltext(
                    # 传入已转义的关键词 tsquery terms。
                    query_words=tsquery_terms,
                    # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
                    top_k=top_k,
                    # 只搜索激活状态的 memory。
                    status="activated",
                    # 限定当前全文检索的 memory scope。
                    scope=scope,
                    # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
                    search_filter=None,
                    # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
                    filter=search_filter,
                    # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
                    user_name=user_name,
                    # 使用 jiebaqry 配置，支持中文分词全文检索。
                    tsquery_config="jiebaqry",
                )
            # 兜底捕获异常，当前实现选择记录日志并降级处理。
            except Exception:
                # 记录非致命异常或配置问题，不中断主搜索流程。
                logger.warning(
                    # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
                    f"[PATH-KEYWORD] search_by_fulltext failed, scope={scope}, user_name={user_name}"
                )
                # 全文检索失败时降级为空结果，避免整个搜索失败。
                hits = []
            # 遍历 fulltext 命中的节点摘要。
            for h in hits or []:
                # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
                hid = str(h.get("id") or "").strip().strip("'\"")
                # 空 id 没有读取节点的意义，跳过。
                if not hid:
                    # 跳过当前候选，继续处理后续数据。
                    continue
                # 读取 fulltext 得分。
                score = h.get("score", 0.0)
                # 同一节点多次命中时只保留最高全文得分。
                if hid not in id_to_score or score > id_to_score[hid]:
                    # 更新该节点 id 的最佳 fulltext 分数。
                    id_to_score[hid] = score
        # 没有任何全文命中时直接返回空结果。
        if not id_to_score:
            # 当前分支没有可用结果时返回空列表，让上层可以安全 extend。
            return []

        # 按 fulltext 分数从高到低排序节点 id。
        sorted_ids = sorted(id_to_score.keys(), key=lambda x: id_to_score[x], reverse=True)
        # 只保留 top_k 个 fulltext 命中 id。
        sorted_ids = sorted_ids[:top_k]
        # 批量读取命中节点的完整内容。
        node_dicts = (
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            self.graph_store.get_nodes(sorted_ids, include_embedding=True, user_name=user_name)
            or []
        )
        # 建立 id 到节点字典，便于按 fulltext 排序恢复原顺序。
        id_to_node = {n.get("id"): n for n in node_dicts}
        # 保存按 fulltext 分数排序后的完整节点。
        ordered_nodes = []

        # 按排序后的 id 顺序重建节点列表。
        for rid in sorted_ids:
            # 只处理成功读到完整节点的 id。
            if rid in id_to_node:
                # 复制节点，避免把 keyword_score 写回原始缓存对象。
                node = copy.deepcopy(id_to_node[rid])
                # 确保节点有 metadata 容器。
                meta = node.setdefault("metadata", {})
                # 默认把 keyword_score 写在 metadata 顶层。
                meta_target = meta
                # 兼容 metadata 内部再包一层 metadata 的结构。
                if isinstance(meta, dict) and isinstance(meta.get("metadata"), dict):
                    # 如果存在嵌套 metadata，就把分数写入真正的内部 metadata。
                    meta_target = meta["metadata"]
                # 根据当前配置、输入参数或中间结果选择是否进入该分支。
                if isinstance(meta_target, dict):
                    # 把 fulltext 分数写入 metadata，方便后续分析或 rerank 使用。
                    meta_target["keyword_score"] = id_to_score[rid]
                # 把处理后的节点加入有序列表。
                ordered_nodes.append(node)

        # 将图数据库节点 dict 转成 TextualMemoryItem。
        results = [TextualMemoryItem.from_dict(n) for n in ordered_nodes]
        # 当前路径召回完成后，按 rerank 开关决定是否重排并返回。
        return self._maybe_rerank(
            rerank,
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            query=query,
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            query_embedding=self._query_embedding_for_rerank(rerank, query_embedding),
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            graph_results=results,
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            top_k=top_k,
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            parsed_goal=parsed_goal,
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            search_filter=search_filter,
        )

    # --- Path B
    # 用 timed 装饰当前阶段，方便日志或监控统计耗时。
    @timed
    # Path B：从 LongTermMemory、UserMemory 或 RawFileMemory 召回长期/用户/文件记忆。
    def _retrieve_from_long_term_and_user(
        self,
        query,
        parsed_goal,
        query_embedding,
        top_k,
        memory_type,
        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
        search_filter: dict | None = None,
        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
        search_priority: dict | None = None,
        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
        user_name: str | None = None,
        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
        id_filter: dict | None = None,
        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
        mode: str = "fast",
        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
        rerank: bool = True,
    ):
        """Retrieve and rerank from LongTermMemory and UserMemory"""
        # 初始化当前路径的结果列表。
        results = []
        # 初始化当前路径内部的并发任务列表。
        tasks = []

        # chain of thinking
        # 初始化 COT embedding 列表，用于复杂问题的多 query 召回。
        cot_embeddings = []
        # 启用 vec_cot 时，会把 query 拆成子问题并分别做 embedding 召回。
        if self.vec_cot:
            # 用 LLM/模板判断是否需要把复杂 query 拆成多个子问题。
            queries = self._cot_query(query, mode=mode, context=parsed_goal.context)
            # 只有拆出了多个子问题时才额外生成 COT embeddings。
            if len(queries) > 1:
                # 为拆解后的子问题生成 embeddings。
                cot_embeddings = self.embedder.embed(queries)
            # 把原始 query embedding 也加入召回向量集合，避免只依赖子问题。
            cot_embeddings.extend(query_embedding)
        # 没有启用相关策略或不满足条件时走默认分支。
        else:
            # 未启用 COT 时，直接使用原始 query embedding。
            cot_embeddings = query_embedding

        # 在长期/用户记忆路径内部并发检索不同 memory scope。
        with ContextThreadPoolExecutor(max_workers=3) as executor:
            # 根据当前配置、输入参数或中间结果选择是否进入该分支。
            if memory_type in ["All", "AllSummaryMemory", "LongTermMemory"]:
                tasks.append(
                    executor.submit(
                        self.graph_retriever.retrieve,
                        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
                        query=query,
                        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
                        parsed_goal=parsed_goal,
                        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
                        query_embedding=cot_embeddings,
                        # 先扩大召回数量，为后续 rerank/去重保留足够候选。
                        top_k=top_k * 2,
                        # 指定该子任务召回 LongTermMemory 范围。
                        memory_scope="LongTermMemory",
                        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
                        search_filter=search_filter,
                        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
                        search_priority=search_priority,
                        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
                        user_name=user_name,
                        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
                        id_filter=id_filter,
                        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
                        use_fast_graph=self.use_fast_graph,
                    )
                )
            # 根据当前配置、输入参数或中间结果选择是否进入该分支。
            if memory_type in ["All", "AllSummaryMemory", "UserMemory"]:
                tasks.append(
                    executor.submit(
                        self.graph_retriever.retrieve,
                        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
                        query=query,
                        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
                        parsed_goal=parsed_goal,
                        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
                        query_embedding=cot_embeddings,
                        # 先扩大召回数量，为后续 rerank/去重保留足够候选。
                        top_k=top_k * 2,
                        # 指定该子任务召回 UserMemory 范围。
                        memory_scope="UserMemory",
                        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
                        search_filter=search_filter,
                        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
                        search_priority=search_priority,
                        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
                        user_name=user_name,
                        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
                        id_filter=id_filter,
                        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
                        use_fast_graph=self.use_fast_graph,
                    )
                )
            # 根据当前配置、输入参数或中间结果选择是否进入该分支。
            if memory_type in ["RawFileMemory"]:
                tasks.append(
                    executor.submit(
                        self.graph_retriever.retrieve,
                        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
                        query=query,
                        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
                        parsed_goal=parsed_goal,
                        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
                        query_embedding=cot_embeddings,
                        # 先扩大召回数量，为后续 rerank/去重保留足够候选。
                        top_k=top_k * 2,
                        # 指定该子任务召回 RawFileMemory 范围。
                        memory_scope="RawFileMemory",
                        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
                        search_filter=search_filter,
                        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
                        search_priority=search_priority,
                        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
                        user_name=user_name,
                        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
                        id_filter=id_filter,
                        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
                        use_fast_graph=self.use_fast_graph,
                    )
                )

            # Collect results from all tasks
            # 遍历内部并发任务，收集各 scope 的结果。
            for task in tasks:
                # 合并某个 scope 的检索结果。
                results.extend(task.result())
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            results = self._deduplicate_rawfile_results(results, user_name=user_name)
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            results = self._filter_intermediate_content(results)

        # 当前路径召回完成后，按 rerank 开关决定是否重排并返回。
        return self._maybe_rerank(
            rerank,
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            query=query,
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            query_embedding=self._query_embedding_for_rerank(rerank, query_embedding),
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            graph_results=results,
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            top_k=top_k,
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            parsed_goal=parsed_goal,
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            search_filter=search_filter,
        )

    # 用 timed 装饰当前阶段，方便日志或监控统计耗时。
    @timed
    # 从指定 cube 中按 embedding 召回 LongTermMemory，常用于跨 cube 或指定 cube 检索。
    def _retrieve_from_memcubes(
        self,
        query,
        parsed_goal,
        query_embedding,
        top_k,
        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
        cube_name="memos_cube01",
        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
        rerank: bool = True,
    ):
        """Retrieve and rerank from LongTermMemory and UserMemory"""
        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
        results = self.graph_retriever.retrieve_from_cube(
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            query_embedding=query_embedding,
            # 先扩大召回数量，为后续 rerank/去重保留足够候选。
            top_k=top_k * 2,
            # 指定该子任务召回 LongTermMemory 范围。
            memory_scope="LongTermMemory",
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            cube_name=cube_name,
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            user_name=cube_name,
        )
        # 当前路径召回完成后，按 rerank 开关决定是否重排并返回。
        return self._maybe_rerank(
            rerank,
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            query=query,
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            query_embedding=self._query_embedding_for_rerank(rerank, query_embedding),
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            graph_results=results,
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            top_k=top_k,
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            parsed_goal=parsed_goal,
        )

    # --- Path C
    # 用 timed 装饰当前阶段，方便日志或监控统计耗时。
    @timed
    # Path C：根据策略和任务解析结果决定是否从互联网检索，并对外部结果 rerank。
    def _retrieve_from_internet(
        self,
        query,
        parsed_goal,
        query_embedding,
        top_k,
        info,
        mode,
        memory_type,
        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
        user_id: str | None = None,
        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
        rerank: bool = True,
    ):
        """Retrieve and rerank from Internet source"""
        # 根据当前配置、输入参数或中间结果选择是否进入该分支。
        if not self.internet_retriever:
            # 调用相关组件完成当前子步骤，并把结果交给后续逻辑。
            logger.info(f"[PATH-C] '{query}' Skipped (no retriever)")
            # 当前分支没有可用结果时返回空列表，让上层可以安全 extend。
            return []
        # 根据当前配置、输入参数或中间结果选择是否进入该分支。
        if self.manual_close_internet and not parsed_goal.internet_search:
            # 调用相关组件完成当前子步骤，并把结果交给后续逻辑。
            logger.info(f"[PATH-C] '{query}' Skipped (no retriever, fast mode)")
            # 当前分支没有可用结果时返回空列表，让上层可以安全 extend。
            return []
        # 互联网路径只在 All 或 OuterMemory 范围内参与召回。
        if memory_type not in ["All", "OuterMemory"]:
            # 调用相关组件完成当前子步骤，并把结果交给后续逻辑。
            logger.info(f"[PATH-C] '{query}' Skipped (memory_type does not match)")
            # 当前分支没有可用结果时返回空列表，让上层可以安全 extend。
            return []
        # 调用相关组件完成当前子步骤，并把结果交给后续逻辑。
        logger.info(f"[PATH-C] '{query}' Retrieving from internet...")
        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
        items = self.internet_retriever.retrieve_from_internet(
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            query=query, top_k=2 * top_k, parsed_goal=parsed_goal, info=info, mode=mode
        )
        # 调用相关组件完成当前子步骤，并把结果交给后续逻辑。
        logger.info(f"[PATH-C] '{query}' Retrieved from internet {len(items)} items: {items}")
        # 当前路径召回完成后，按 rerank 开关决定是否重排并返回。
        return self._maybe_rerank(
            rerank,
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            query=query,
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            query_embedding=self._query_embedding_for_rerank(rerank, query_embedding),
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            graph_results=items,
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            top_k=top_k,
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            parsed_goal=parsed_goal,
        )

    # --- Path D
    # 用 timed 装饰当前阶段，方便日志或监控统计耗时。
    @timed
    # Path D：并行召回 ToolSchemaMemory 和 ToolTrajectoryMemory，再分别 rerank 合并。
    def _retrieve_from_tool_memory(
        self,
        query,
        parsed_goal,
        query_embedding,
        top_k,
        memory_type,
        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
        search_filter: dict | None = None,
        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
        search_priority: dict | None = None,
        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
        user_name: str | None = None,
        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
        id_filter: dict | None = None,
        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
        mode: str = "fast",
        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
        rerank: bool = True,
    ):
        """Retrieve and rerank from ToolMemory"""
        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
        results = {
            "ToolSchemaMemory": [],
            "ToolTrajectoryMemory": [],
        }
        # 初始化当前路径内部的并发任务列表。
        tasks = []

        # chain of thinking
        # 初始化 COT embedding 列表，用于复杂问题的多 query 召回。
        cot_embeddings = []
        # 启用 vec_cot 时，会把 query 拆成子问题并分别做 embedding 召回。
        if self.vec_cot:
            # 用 LLM/模板判断是否需要把复杂 query 拆成多个子问题。
            queries = self._cot_query(query, mode=mode, context=parsed_goal.context)
            # 只有拆出了多个子问题时才额外生成 COT embeddings。
            if len(queries) > 1:
                # 为拆解后的子问题生成 embeddings。
                cot_embeddings = self.embedder.embed(queries)
            # 把原始 query embedding 也加入召回向量集合，避免只依赖子问题。
            cot_embeddings.extend(query_embedding)
        # 没有启用相关策略或不满足条件时走默认分支。
        else:
            # 未启用 COT 时，直接使用原始 query embedding。
            cot_embeddings = query_embedding

        # 在工具记忆路径内部并发检索 schema 和 trajectory 两类工具记忆。
        with ContextThreadPoolExecutor(max_workers=2) as executor:
            # 根据当前配置、输入参数或中间结果选择是否进入该分支。
            if memory_type in ["All", "ToolSchemaMemory"]:
                tasks.append(
                    executor.submit(
                        self.graph_retriever.retrieve,
                        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
                        query=query,
                        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
                        parsed_goal=parsed_goal,
                        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
                        query_embedding=cot_embeddings,
                        # 先扩大召回数量，为后续 rerank/去重保留足够候选。
                        top_k=top_k * 2,
                        # 指定该子任务召回工具 schema 记忆。
                        memory_scope="ToolSchemaMemory",
                        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
                        search_filter=search_filter,
                        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
                        search_priority=search_priority,
                        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
                        user_name=user_name,
                        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
                        id_filter=id_filter,
                        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
                        use_fast_graph=self.use_fast_graph,
                    )
                )
            # 根据当前配置、输入参数或中间结果选择是否进入该分支。
            if memory_type in ["All", "ToolTrajectoryMemory"]:
                tasks.append(
                    executor.submit(
                        self.graph_retriever.retrieve,
                        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
                        query=query,
                        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
                        parsed_goal=parsed_goal,
                        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
                        query_embedding=cot_embeddings,
                        # 先扩大召回数量，为后续 rerank/去重保留足够候选。
                        top_k=top_k * 2,
                        # 指定该子任务召回工具调用轨迹记忆。
                        memory_scope="ToolTrajectoryMemory",
                        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
                        search_filter=search_filter,
                        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
                        search_priority=search_priority,
                        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
                        user_name=user_name,
                        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
                        id_filter=id_filter,
                        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
                        use_fast_graph=self.use_fast_graph,
                    )
                )

            # Collect results from all tasks
            # 遍历内部并发任务，收集各 scope 的结果。
            for task in tasks:
                # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
                rsp = task.result()
                # 根据当前配置、输入参数或中间结果选择是否进入该分支。
                if rsp and rsp[0].metadata.memory_type == "ToolSchemaMemory":
                    # 调用相关组件完成当前子步骤，并把结果交给后续逻辑。
                    results["ToolSchemaMemory"].extend(rsp)
                # 在前一条件不满足时，继续检查这个替代条件。
                elif rsp and rsp[0].metadata.memory_type == "ToolTrajectoryMemory":
                    # 调用相关组件完成当前子步骤，并把结果交给后续逻辑。
                    results["ToolTrajectoryMemory"].extend(rsp)

        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
        schema_reranked = self._maybe_rerank(
            rerank,
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            query=query,
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            query_embedding=self._query_embedding_for_rerank(rerank, query_embedding),
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            graph_results=results["ToolSchemaMemory"],
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            top_k=top_k,
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            parsed_goal=parsed_goal,
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            search_filter=search_filter,
        )
        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
        trajectory_reranked = self._maybe_rerank(
            rerank,
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            query=query,
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            query_embedding=self._query_embedding_for_rerank(rerank, query_embedding),
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            graph_results=results["ToolTrajectoryMemory"],
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            top_k=top_k,
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            parsed_goal=parsed_goal,
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            search_filter=search_filter,
        )
        # 返回当前步骤的处理结果，供上层搜索流程继续使用。
        return schema_reranked + trajectory_reranked

    # --- Path E
    # 用 timed 装饰当前阶段，方便日志或监控统计耗时。
    @timed
    # Path E：从 SkillMemory 中召回技能类记忆并 rerank。
    def _retrieve_from_skill_memory(
        self,
        query,
        parsed_goal,
        query_embedding,
        top_k,
        memory_type,
        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
        search_filter: dict | None = None,
        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
        search_priority: dict | None = None,
        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
        user_name: str | None = None,
        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
        id_filter: dict | None = None,
        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
        mode: str = "fast",
        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
        rerank: bool = True,
    ):
        """Retrieve and rerank from SkillMemory"""

        # SkillMemory 路径只在请求允许技能记忆时执行。
        if memory_type not in ["All", "SkillMemory"]:
            # 调用相关组件完成当前子步骤，并把结果交给后续逻辑。
            logger.info(f"[PATH-E] '{query}' Skipped (memory_type does not match)")
            # 当前分支没有可用结果时返回空列表，让上层可以安全 extend。
            return []

        # chain of thinking
        # 初始化 COT embedding 列表，用于复杂问题的多 query 召回。
        cot_embeddings = []
        # 启用 vec_cot 时，会把 query 拆成子问题并分别做 embedding 召回。
        if self.vec_cot:
            # 用 LLM/模板判断是否需要把复杂 query 拆成多个子问题。
            queries = self._cot_query(query, mode=mode, context=parsed_goal.context)
            # 只有拆出了多个子问题时才额外生成 COT embeddings。
            if len(queries) > 1:
                # 为拆解后的子问题生成 embeddings。
                cot_embeddings = self.embedder.embed(queries)
            # 把原始 query embedding 也加入召回向量集合，避免只依赖子问题。
            cot_embeddings.extend(query_embedding)
        # 没有启用相关策略或不满足条件时走默认分支。
        else:
            # 未启用 COT 时，直接使用原始 query embedding。
            cot_embeddings = query_embedding

        # 调用 GraphMemoryRetriever 从图数据库按指定 memory_scope 召回节点。
        items = self.graph_retriever.retrieve(
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            query=query,
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            parsed_goal=parsed_goal,
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            query_embedding=cot_embeddings,
            # 先扩大召回数量，为后续 rerank/去重保留足够候选。
            top_k=top_k * 2,
            # 指定当前路径召回技能记忆。
            memory_scope="SkillMemory",
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            search_filter=search_filter,
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            search_priority=search_priority,
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            user_name=user_name,
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            id_filter=id_filter,
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            use_fast_graph=self.use_fast_graph,
        )

        # 当前路径召回完成后，按 rerank 开关决定是否重排并返回。
        return self._maybe_rerank(
            rerank,
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            query=query,
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            query_embedding=self._query_embedding_for_rerank(rerank, query_embedding),
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            graph_results=items,
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            top_k=top_k,
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            parsed_goal=parsed_goal,
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            search_filter=search_filter,
        )

    # 用 timed 装饰当前阶段，方便日志或监控统计耗时。
    @timed
    # Path F：从 PreferenceMemory 中召回偏好记忆并 rerank。
    def _retrieve_from_preference_memory(
        self,
        query,
        parsed_goal,
        query_embedding,
        top_k,
        memory_type,
        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
        search_filter: dict | None = None,
        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
        search_priority: dict | None = None,
        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
        user_name: str | None = None,
        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
        id_filter: dict | None = None,
        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
        mode: str = "fast",
        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
        rerank: bool = True,
    ):
        """Retrieve and rerank from PreferenceMemory"""
        # PreferenceMemory 路径只在请求允许偏好记忆时执行。
        if memory_type not in ["All", "PreferenceMemory"]:
            # 调用相关组件完成当前子步骤，并把结果交给后续逻辑。
            logger.info(f"[PATH-F] '{query}' Skipped (memory_type does not match)")
            # 当前分支没有可用结果时返回空列表，让上层可以安全 extend。
            return []

        # chain of thinking
        # 初始化 COT embedding 列表，用于复杂问题的多 query 召回。
        cot_embeddings = []
        # 启用 vec_cot 时，会把 query 拆成子问题并分别做 embedding 召回。
        if self.vec_cot:
            # 用 LLM/模板判断是否需要把复杂 query 拆成多个子问题。
            queries = self._cot_query(query, mode=mode, context=parsed_goal.context)
            # 只有拆出了多个子问题时才额外生成 COT embeddings。
            if len(queries) > 1:
                # 为拆解后的子问题生成 embeddings。
                cot_embeddings = self.embedder.embed(queries)
            # 把原始 query embedding 也加入召回向量集合，避免只依赖子问题。
            cot_embeddings.extend(query_embedding)
        # 没有启用相关策略或不满足条件时走默认分支。
        else:
            # 未启用 COT 时，直接使用原始 query embedding。
            cot_embeddings = query_embedding

        # 调用 GraphMemoryRetriever 从图数据库按指定 memory_scope 召回节点。
        items = self.graph_retriever.retrieve(
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            query=query,
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            parsed_goal=parsed_goal,
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            query_embedding=cot_embeddings,
            # 先扩大召回数量，为后续 rerank/去重保留足够候选。
            top_k=top_k * 2,
            # 指定当前路径召回偏好记忆。
            memory_scope="PreferenceMemory",
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            search_filter=search_filter,
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            search_priority=search_priority,
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            user_name=user_name,
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            id_filter=id_filter,
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            use_fast_graph=self.use_fast_graph,
        )

        # 当前路径召回完成后，按 rerank 开关决定是否重排并返回。
        return self._maybe_rerank(
            rerank,
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            query=query,
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            query_embedding=self._query_embedding_for_rerank(rerank, query_embedding),
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            graph_results=items,
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            top_k=top_k,
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            parsed_goal=parsed_goal,
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            search_filter=search_filter,
        )

    # 用 timed 装饰当前阶段，方便日志或监控统计耗时。
    @timed
    # plugin 模式的简化检索路径：用 query 和分词做混合召回，再用相似度挑选不重复结果。
    def _retrieve_simple(
        self,
        query: str,
        top_k: int,
        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
        search_filter: dict | None = None,
        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
        user_name: str | None = None,
        **kwargs,
    ):
        """
        Retrieve from by keywords and embedding, this func is hotfix for sources=plugin mode
        will merge with fulltext retrieval in the future
        """
        # 初始化简化检索的 query words。
        query_words = []
        # 如果配置了 tokenizer，就使用更可靠的混合分词。
        if self.tokenizer:
            # 对中英文混合 query 分词。
            query_words = self.tokenizer.tokenize_mixed(query)
        # 没有 tokenizer 时使用最简单的空格切分兜底。
        else:
            # 用空格拆 query，适合英文或已分词文本。
            query_words = query.strip().split()
        # 对分词结果去重并限制数量，避免 embedding 输入过多。
        query_words = list(set(query_words))[: top_k * 3]
        # 保留完整 query，同时加入拆分词，形成混合检索输入。
        query_words = [query, *query_words]
        # 调用相关组件完成当前子步骤，并把结果交给后续逻辑。
        logger.info(f"[SIMPLESEARCH] Query words: {query_words}")
        # 为完整 query 和关键词生成 embeddings。
        query_embeddings = self.embedder.embed(query_words)

        # 调用混合检索接口召回候选节点。
        items = self.graph_retriever.retrieve_from_mixed(
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            top_k=top_k * 2,
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            memory_scope=None,
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            query_embedding=query_embeddings,
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            search_filter=search_filter,
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            user_name=user_name,
        )
        # 调用相关组件完成当前子步骤，并把结果交给后续逻辑。
        logger.info(f"[SIMPLESEARCH] Items count: {len(items)}")
        # 提取候选 memory 文本，用于计算文档间相似度。
        documents = [getattr(item, "memory", "") for item in items]
        # 没有文档内容时无法继续相似度筛选。
        if not documents:
            # 当前分支没有可用结果时返回空列表，让上层可以安全 extend。
            return []
        # 为候选文档生成 embeddings。
        documents_embeddings = self.embedder.embed(documents)
        # embedding 结果为空时直接返回空。
        if not documents_embeddings:
            # 调用相关组件完成当前子步骤，并把结果交给后续逻辑。
            logger.info("[SIMPLESEARCH] Documents embeddings is empty")
            # 当前分支没有可用结果时返回空列表，让上层可以安全 extend。
            return []
        # 计算候选文档之间的相似度矩阵。
        similarity_matrix = cosine_similarity_matrix(documents_embeddings)
        # 挑出互相不太相关的一组候选，减少重复结果。
        selected_indices, _ = find_best_unrelated_subgroup(documents, similarity_matrix)
        # 根据选中下标恢复 TextualMemoryItem。
        selected_items = [items[i] for i in selected_indices]
        # 记录关键运行状态，方便追踪搜索路径、候选数量或分支选择。
        logger.info(
            f"[SIMPLESEARCH] after unrelated subgroup selection items count: {len(selected_items)}"
        )
        # 简化路径同样支持 rerank 开关，默认开启。
        rerank = bool(kwargs.get("rerank", True))
        # 返回当前步骤的处理结果，供上层搜索流程继续使用。
        return self._maybe_rerank(
            rerank,
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            query=query,
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            query_embedding=self._query_embedding_for_rerank(rerank, query_embeddings),
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            graph_results=selected_items,
            # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
            top_k=top_k,
        )

    # 用 timed 装饰当前阶段，方便日志或监控统计耗时。
    @timed
    # 按 memory 文本去重，同文本只保留分数最高的候选。
    def _deduplicate_results(self, results):
        """Deduplicate results by memory text"""
        # 用 memory 文本作为 key 保存最佳候选。
        deduped = {}
        # 遍历所有带分数的候选。
        for item, score in results:
            # 同文本只保留分数更高的一条。
            if item.memory not in deduped or score > deduped[item.memory][1]:
                # 更新该 memory 文本对应的最佳候选。
                deduped[item.memory] = (item, score)
        # 返回去重后的候选列表。
        return list(deduped.values())

    # 用 timed 装饰当前阶段，方便日志或监控统计耗时。
    @timed
    # 按记忆类型分组排序和截断，把分数写入 metadata.relativity 后生成最终 TextualMemoryItem。
    def _sort_and_trim(
        self,
        results,
        top_k,
        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
        plugin=False,
        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
        search_tool_memory=False,
        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
        tool_mem_top_k=6,
        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
        include_skill_memory=False,
        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
        skill_mem_top_k=3,
        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
        include_preference_memory=False,
        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
        pref_mem_top_k=6,
    ):
        """Sort results by score and trim to top_k"""
        # 收集最终返回的 TextualMemoryItem。
        final_items = []
        # 如果启用工具记忆返回，先单独处理工具 schema/trajectory。
        if search_tool_memory:
            # 筛出 ToolSchemaMemory 候选。
            tool_schema_results = [
                # 调用相关组件完成当前子步骤，并把结果交给后续逻辑。
                (item, score)
                # 遍历当前集合，逐项累积结果或执行过滤逻辑。
                for item, score in results
                # 根据当前配置、输入参数或中间结果选择是否进入该分支。
                if item.metadata.memory_type == "ToolSchemaMemory"
            ]
            # 按得分排序工具 schema 候选。
            sorted_tool_schema_results = sorted(
                # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
                tool_schema_results, key=lambda pair: pair[1], reverse=True
            )[:tool_mem_top_k]
            # 遍历排序后的工具 schema 结果。
            for item, score in sorted_tool_schema_results:
                # plugin 模式下过滤掉近似 0 分的弱相关结果。
                if plugin and round(score, 2) == 0.00:
                    # 跳过当前候选，继续处理后续数据。
                    continue
                # 复制原 metadata 为 dict，方便写入 relativity 分数。
                meta_data = item.metadata.model_dump()
                # 把当前 rerank/召回分数写入 relativity 字段。
                meta_data["relativity"] = score
                # 构造新的最终 item 并加入返回列表。
                final_items.append(
                    # 重新构造 TextualMemoryItem，保持 id/memory，同时替换成搜索结果 metadata。
                    TextualMemoryItem(
                        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
                        id=item.id,
                        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
                        memory=item.memory,
                        # 用带 relativity 的 metadata dict 构造搜索结果专用 metadata。
                        metadata=SearchedTreeNodeTextualMemoryMetadata(**meta_data),
                    )
                )
            # 筛出 ToolTrajectoryMemory 候选。
            tool_trajectory_results = [
                # 调用相关组件完成当前子步骤，并把结果交给后续逻辑。
                (item, score)
                # 遍历当前集合，逐项累积结果或执行过滤逻辑。
                for item, score in results
                # 根据当前配置、输入参数或中间结果选择是否进入该分支。
                if item.metadata.memory_type == "ToolTrajectoryMemory"
            ]
            # 按得分排序工具轨迹候选。
            sorted_tool_trajectory_results = sorted(
                # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
                tool_trajectory_results, key=lambda pair: pair[1], reverse=True
            )[:tool_mem_top_k]
            # 遍历排序后的工具轨迹结果。
            for item, score in sorted_tool_trajectory_results:
                # plugin 模式下过滤掉近似 0 分的弱相关结果。
                if plugin and round(score, 2) == 0.00:
                    # 跳过当前候选，继续处理后续数据。
                    continue
                # 复制原 metadata 为 dict，方便写入 relativity 分数。
                meta_data = item.metadata.model_dump()
                # 把当前 rerank/召回分数写入 relativity 字段。
                meta_data["relativity"] = score
                # 构造新的最终 item 并加入返回列表。
                final_items.append(
                    # 重新构造 TextualMemoryItem，保持 id/memory，同时替换成搜索结果 metadata。
                    TextualMemoryItem(
                        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
                        id=item.id,
                        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
                        memory=item.memory,
                        # 用带 relativity 的 metadata dict 构造搜索结果专用 metadata。
                        metadata=SearchedTreeNodeTextualMemoryMetadata(**meta_data),
                    )
                )

        # 如果请求技能记忆，则单独处理 SkillMemory。
        if include_skill_memory:
            # 筛出技能记忆候选。
            skill_results = [
                # 调用相关组件完成当前子步骤，并把结果交给后续逻辑。
                (item, score)
                # 遍历当前集合，逐项累积结果或执行过滤逻辑。
                for item, score in results
                # 根据当前配置、输入参数或中间结果选择是否进入该分支。
                if item.metadata.memory_type == "SkillMemory"
            ]
            # 按得分排序技能记忆候选，并准备截断。
            sorted_skill_results = sorted(skill_results, key=lambda pair: pair[1], reverse=True)[
                :skill_mem_top_k
            ]
            # 遍历排序后的技能记忆结果。
            for item, score in sorted_skill_results:
                # plugin 模式下过滤掉近似 0 分的弱相关结果。
                if plugin and round(score, 2) == 0.00:
                    # 跳过当前候选，继续处理后续数据。
                    continue
                # 复制原 metadata 为 dict，方便写入 relativity 分数。
                meta_data = item.metadata.model_dump()
                # 把当前 rerank/召回分数写入 relativity 字段。
                meta_data["relativity"] = score
                # 构造新的最终 item 并加入返回列表。
                final_items.append(
                    # 重新构造 TextualMemoryItem，保持 id/memory，同时替换成搜索结果 metadata。
                    TextualMemoryItem(
                        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
                        id=item.id,
                        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
                        memory=item.memory,
                        # 用带 relativity 的 metadata dict 构造搜索结果专用 metadata。
                        metadata=SearchedTreeNodeTextualMemoryMetadata(**meta_data),
                    )
                )

        # 如果请求偏好记忆，则单独处理 PreferenceMemory。
        if include_preference_memory:
            # 筛出偏好记忆候选。
            pref_results = [
                # 调用相关组件完成当前子步骤，并把结果交给后续逻辑。
                (item, score)
                # 遍历当前集合，逐项累积结果或执行过滤逻辑。
                for item, score in results
                # 根据当前配置、输入参数或中间结果选择是否进入该分支。
                if item.metadata.memory_type == "PreferenceMemory"
            ]
            # 按得分排序偏好记忆候选，并准备截断。
            sorted_pref_results = sorted(pref_results, key=lambda pair: pair[1], reverse=True)[
                :pref_mem_top_k
            ]
            # 遍历排序后的偏好记忆结果。
            for item, score in sorted_pref_results:
                # plugin 模式下过滤掉近似 0 分的弱相关结果。
                if plugin and round(score, 2) == 0.00:
                    # 跳过当前候选，继续处理后续数据。
                    continue
                # 复制原 metadata 为 dict，方便写入 relativity 分数。
                meta_data = item.metadata.model_dump()
                # 把当前 rerank/召回分数写入 relativity 字段。
                meta_data["relativity"] = score
                # 构造新的最终 item 并加入返回列表。
                final_items.append(
                    # 重新构造 TextualMemoryItem，保持 id/memory，同时替换成搜索结果 metadata。
                    TextualMemoryItem(
                        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
                        id=item.id,
                        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
                        memory=item.memory,
                        # 用带 relativity 的 metadata dict 构造搜索结果专用 metadata。
                        metadata=SearchedTreeNodeTextualMemoryMetadata(**meta_data),
                    )
                )

        # separate textual results
        # 从剩余候选中筛出普通文本/文件记忆类型。
        results = [
            # 调用相关组件完成当前子步骤，并把结果交给后续逻辑。
            (item, score)
            # 遍历当前集合，逐项累积结果或执行过滤逻辑。
            for item, score in results
            # 根据当前配置、输入参数或中间结果选择是否进入该分支。
            if item.metadata.memory_type
            in ["WorkingMemory", "LongTermMemory", "UserMemory", "OuterMemory", "RawFileMemory"]
        ]

        # 普通文本结果按分数降序排序，并截断到 top_k。
        sorted_results = sorted(results, key=lambda pair: pair[1], reverse=True)[:top_k]

        # 遍历最终普通文本候选。
        for item, score in sorted_results:
            # plugin 模式下过滤掉近似 0 分的弱相关结果。
            if plugin and round(score, 2) == 0.00:
                # 跳过当前候选，继续处理后续数据。
                continue
            # 复制原 metadata 为 dict，方便写入 relativity 分数。
            meta_data = item.metadata.model_dump()
            # 把当前 rerank/召回分数写入 relativity 字段。
            meta_data["relativity"] = score
            # 构造新的最终 item 并加入返回列表。
            final_items.append(
                # 重新构造 TextualMemoryItem，保持 id/memory，同时替换成搜索结果 metadata。
                TextualMemoryItem(
                    # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
                    id=item.id,
                    # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
                    memory=item.memory,
                    # 用带 relativity 的 metadata dict 构造搜索结果专用 metadata。
                    metadata=SearchedTreeNodeTextualMemoryMetadata(**meta_data),
                )
            )
        # 返回合并后的工具、技能、偏好和普通记忆结果。
        return final_items

    # 用 timed 装饰当前阶段，方便日志或监控统计耗时。
    @timed
    # RawFileMemory 去重：如果 rawfile 指向 summary 节点，就从结果里移除被指向的 summary。
    def _deduplicate_rawfile_results(self, results, user_name: str | None = None):
        """
        Deduplicate rawfile related memories by edge
        """
        # 没有候选时无需做 rawfile 去重。
        if not results:
            # 返回当前阶段处理后的结果，交给上层继续后处理。
            return results

        # 收集需要从结果中移除的 summary 节点 id。
        summary_ids_to_remove = set()
        # 筛出 RawFileMemory 节点。
        rawfile_items = [item for item in results if item.metadata.memory_type == "RawFileMemory"]
        # 没有 RawFileMemory 时无需按 SUMMARY 边去重。
        if not rawfile_items:
            # 返回当前阶段处理后的结果，交给上层继续后处理。
            return results

        # 并发查询每个 rawfile 的 SUMMARY 边，最多开 10 个线程。
        with ContextThreadPoolExecutor(max_workers=min(len(rawfile_items), 10)) as executor:
            # 提交所有 rawfile edge 查询任务。
            futures = [
                # 把单个 rawfile 的 edge 查询提交到线程池。
                executor.submit(
                    # 从图数据库读取指定节点的边。
                    self.graph_store.get_edges,
                    rawfile_item.id,
                    # 只查询 SUMMARY 类型的边。
                    type="SUMMARY",
                    # 只看 rawfile 指向 summary 的出边。
                    direction="OUTGOING",
                    # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
                    user_name=user_name,
                )
                # 遍历当前集合，逐项累积结果或执行过滤逻辑。
                for rawfile_item in rawfile_items
            ]
            # 按 future 完成顺序处理 edge 查询结果。
            for future in as_completed(futures):
                # 开始保护性执行，避免某个子步骤异常拖垮整条搜索链路。
                try:
                    # 获取某个 rawfile 的 SUMMARY 边列表。
                    edges = future.result()
                    # 遍历边，找出被 rawfile 指向的 summary 节点。
                    for edge in edges:
                        # 读取 SUMMARY 边的目标节点 id。
                        summary_target_id = edge.get("to")
                        # 只有存在目标 id 时才记录。
                        if summary_target_id:
                            # 把 summary 节点加入待移除集合。
                            summary_ids_to_remove.add(summary_target_id)
                            logger.debug(
                                f"[DEDUP] Marking summary node {summary_target_id} for removal (pointed by RawFileMemory)"
                            )
                # 捕获异常对象，便于日志中输出具体错误原因。
                except Exception as e:
                    # 调用相关组件完成当前子步骤，并把结果交给后续逻辑。
                    logger.warning(f"[DEDUP] Failed to get summary target ids: {e}")

        # 保存去除重复 summary 后的结果。
        filtered_results = []
        # 遍历原始候选，决定是否保留。
        for item in results:
            # 如果当前 item 是 rawfile 已覆盖的 summary，则移除。
            if item.id in summary_ids_to_remove:
                logger.debug(
                    f"[DEDUP] Removing summary node {item.id} because it is pointed by RawFileMemory"
                )
                # 跳过当前候选，继续处理后续数据。
                continue
            # 保留非重复候选。
            filtered_results.append(item)

        # 返回 rawfile 去重后的结果。
        return filtered_results

    # 过滤文件上传中间内容，避免把 File URL/File ID/Filename 这类元信息当作有效记忆返回。
    def _filter_intermediate_content(self, results):
        """Filter intermediate content"""
        # 保存过滤后的有效记忆。
        filtered_results = []
        # 遍历召回结果。
        for item in results:
            # 开始判断 memory 内容是否不是文件处理中间元信息。
            if (
                # 排除只包含文件 URL 的中间内容。
                "File URL:" not in item.memory
                # 排除只包含文件 ID 的中间内容。
                and "File ID:" not in item.memory
                # 排除只包含文件名的中间内容。
                and "Filename:" not in item.memory
            ):
                # 保留真正的内容型 memory。
                filtered_results.append(item)
        # 返回过滤后的结果。
        return filtered_results

    # 用 timed 装饰当前阶段，方便日志或监控统计耗时。
    @timed
    # 记录检索结果使用历史的预留逻辑；当前主体代码被三引号注释包住，实际不执行。
    def _update_usage_history(self, items, info, user_name: str | None = None):
        """Update usage history in graph DB
        now_time = datetime.now().isoformat()
        info_copy = dict(info or {})
        info_copy.pop("chat_history", None)
        usage_record = json.dumps({"time": now_time, "info": info_copy})
        payload = []
        for it in items:
            try:
                item_id = getattr(it, "id", None)
                md = getattr(it, "metadata", None)
                if md is None:
                    continue
                if not hasattr(md, "usage") or md.usage is None:
                    md.usage = []
                md.usage.append(usage_record)
                if item_id:
                    payload.append((item_id, list(md.usage)))
            except Exception:
                logger.exception("[USAGE] snapshot item failed")

        if payload:
            self._usage_executor.submit(
                self._update_usage_history_worker, payload, usage_record, user_name
            )
        """

    # 后台写入 usage history 到图数据库节点。
    def _update_usage_history_worker(
        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
        self, payload, usage_record: str, user_name: str | None = None
    ):
        # 开始保护性执行，避免某个子步骤异常拖垮整条搜索链路。
        try:
            # 遍历待更新的节点 id 和 usage 列表。
            for item_id, usage_list in payload:
                # 更新图数据库中该 memory 节点的 usage 字段。
                self.graph_store.update_node(item_id, {"usage": usage_list}, user_name=user_name)
        # 兜底捕获异常，当前实现选择记录日志并降级处理。
        except Exception:
            # 记录 usage 写入失败的完整堆栈。
            logger.exception("[USAGE] update usage failed")

    # 用 LLM 判断 query 是否复杂；复杂时拆成多个子问题作为 COT 检索 query。
    def _cot_query(
        self,
        query,
        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
        mode="fast",
        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
        split_num: int = 3,
        # 计算或保存当前步骤的中间变量，供后续检索、排序或过滤使用。
        context: list[str] | None = None,
    ) -> list[str]:
        """Generate chain-of-thought queries"""

        # 检测 query 语言，用来选择中文或英文 COT prompt 模板。
        lang = detect_lang(query)
        # fine 模式且有上下文时使用包含 context 的完整拆解模板。
        if mode == "fine" and context:
            # 选择 fine 模式下对应语言的 prompt。
            template = COT_DICT["fine"][lang]
            # 开始把原始 query、拆分数量和上下文填入 prompt 模板。
            prompt = (
                # 把模板中的原始 query 占位符替换成当前 query。
                template.replace("${original_query}", query)
                # 把最大拆分数量写入 prompt。
                .replace("${split_num_threshold}", str(split_num))
                # 调用相关组件完成当前子步骤，并把结果交给后续逻辑。
                .replace("${context}", "\n".join(context))
            )
        # fast 模式或无上下文时使用简化 COT prompt。
        else:
            # 选择 fast 模式下对应语言的简化模板。
            template = COT_DICT["fast"][lang]
            # 为简化模板填入原始 query 和拆分数量。
            prompt = template.replace("${original_query}", query).replace(
                # 调用相关组件完成当前子步骤，并把结果交给后续逻辑。
                "${split_num_threshold}", str(split_num)
            )

        # 构造 LLM chat messages，请求模型判断是否需要拆分 query。
        messages = [{"role": "user", "content": prompt}]
        # 开始保护性执行，避免某个子步骤异常拖垮整条搜索链路。
        try:
            # 以确定性参数调用 LLM，生成 JSON 风格的拆解结果。
            response_text = self.llm.generate(messages, temperature=0, top_p=1)
            # 把 LLM 文本解析成 JSON dict。
            response_json = parse_json_result(response_text)
            # 要求返回中必须包含 is_complex 字段。
            assert "is_complex" in response_json
            # 如果模型认为问题不复杂，就不拆分。
            if not response_json["is_complex"]:
                # 异常时降级为只使用原 query，保证搜索仍可继续。
                return [query]
            # fast 模式或无上下文时使用简化 COT prompt。
            else:
                # 复杂问题必须返回 sub_questions 字段。
                assert "sub_questions" in response_json
                # 调用相关组件完成当前子步骤，并把结果交给后续逻辑。
                logger.info("Query: {} COT: {}".format(query, response_json["sub_questions"]))
                # 返回最多 split_num 个子问题。
                return response_json["sub_questions"][:split_num]
        # 捕获异常对象，便于日志中输出具体错误原因。
        except Exception as e:
            # 调用相关组件完成当前子步骤，并把结果交给后续逻辑。
            logger.error(f"[LLM] Exception during chat generation: {e}")
            # 异常时降级为只使用原 query，保证搜索仍可继续。
            return [query]
