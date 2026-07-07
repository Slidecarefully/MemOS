# 引入 concurrent.futures，用于等待并发添加边、并发处理任务等线程池 Future 完成。
import concurrent.futures
# 引入 json，用于从文件加载记忆备份，以及把导出的图数据写成 JSON。
import json
# 引入 os，用于拼接文件路径、判断备份文件是否存在、创建目录等文件系统操作。
import os
# 引入 shutil，用于删除旧备份目录。
import shutil
# 引入 tempfile，用于选择系统临时目录作为数据库 drop 前的备份根目录。
import tempfile
# 引入 time，用于统计 RawFile 边写入耗时。
import time

# 引入 datetime，用于排序 WorkingMemory、生成备份时间戳等时间相关逻辑。
from datetime import datetime
# 引入 Path，用于以更清晰的方式组织备份目录路径。
from pathlib import Path
# 引入 Any 和 Literal，用于类型标注；Literal 限定子图搜索方式只能是 embedding 或 fulltext。
from typing import Any, Literal

# 引入 TreeTextMemory 的配置模型，初始化时会从中读取 LLM、embedder、graph_db、搜索策略等配置。
from memos.configs.memory import TreeTextMemoryConfig
# 引入 reranker 配置工厂，用于在没有显式 reranker 配置时构造默认配置。
from memos.configs.reranker import RerankerConfigFactory
# 引入带上下文传播能力的线程池，保证并发任务中仍能保留请求上下文或日志链路。
from memos.context.context import ContextThreadPoolExecutor
# 引入依赖检查装饰器，用于在中文分词时按需检查 jieba 是否安装。
from memos.dependency import require_python_package
# 引入 embedding 工厂和 OllamaEmbedder 类型；初始化时通过配置创建向量化组件。
from memos.embedders.factory import EmbedderFactory, OllamaEmbedder
# 引入图数据库工厂和 Neo4j 类型；TreeTextMemory 通过它创建和访问图存储。
from memos.graph_dbs.factory import GraphStoreFactory, Neo4jGraphDB
# 引入 LLM 工厂和支持的 LLM 类型，用于创建抽取模型和检索调度模型。
from memos.llms.factory import AzureLLM, LLMFactory, OllamaLLM, OpenAILLM
# 引入统一日志工厂，保证该模块日志格式与项目整体一致。
from memos.log import get_logger
# 引入语言检测函数；全文检索时根据查询语言决定是否使用中文分词。
from memos.mem_reader.read_multi_modal.utils import detect_lang
# 引入文本记忆基类，TreeTextMemory 继承它并实现具体存储、检索和删除逻辑。
from memos.memories.textual.base import BaseTextMemory
# 引入文本记忆数据结构和树/图节点 metadata，用于把图数据库记录还原成记忆对象。
from memos.memories.textual.item import TextualMemoryItem, TreeNodeTextualMemoryMetadata
# 引入 MemoryManager；TreeTextMemory 将实际新增、WorkingMemory 替换和容量统计委托给它。
from memos.memories.textual.tree_text_memory.organize.manager import MemoryManager
# 引入高级搜索器模块；下面会把 AdvancedSearcher 别名为 Searcher。
from memos.memories.textual.tree_text_memory.retrieve.advanced_searcher import (
    # 把 AdvancedSearcher 命名为 Searcher，后续创建搜索器时更简洁。
    AdvancedSearcher as Searcher,
# 结束当前多行调用或结构定义。
)
# 引入 BM25 检索器；当 search_strategy 启用 bm25 时用于关键词召回。
from memos.memories.textual.tree_text_memory.retrieve.bm25_util import EnhancedBM25
# 引入互联网检索器工厂模块；配置开启时用于创建联网检索组件。
from memos.memories.textual.tree_text_memory.retrieve.internet_retriever_factory import (
    # 导入 InternetRetrieverFactory，用于按配置创建具体互联网检索后端。
    InternetRetrieverFactory,
# 结束当前多行调用或结构定义。
)
# 引入停用词管理器；中文全文检索分词后会过滤停用词。
from memos.memories.textual.tree_text_memory.retrieve.retrieve_utils import StopwordManager
# 引入 reranker 工厂，用于创建默认或自定义的重排器。
from memos.reranker.factory import RerankerFactory
# 引入 MessageList 类型，供 extract 抽象方法保持统一接口。
from memos.types import MessageList


# 创建模块级 logger；后续所有告警、调试和错误信息都通过它输出。
logger = get_logger(__name__)


# 定义 TreeTextMemory，它是基于图结构的文本记忆实现，负责统一封装写入、检索、导入导出和删除能力。
class TreeTextMemory(BaseTextMemory):
    """General textual memory implementation for storing and retrieving memories."""

    # 初始化 TreeTextMemory；核心任务是根据配置构造 LLM、向量模型、图数据库、搜索器依赖和 MemoryManager。
    def __init__(self, config: TreeTextMemoryConfig):
        """Initialize memory with the given configuration."""
        # Set mode from class default or override if needed
        # 读取运行模式；后续 add 会把该模式传给 MemoryManager，影响同步/异步清理等行为。
        self.mode = config.mode
        # 记录当前记忆系统模式，方便启动时确认运行方式。
        logger.info(f"Tree mode is {self.mode}")

        # 保存完整配置对象，后续 load/dump 等方法还会读取配置中的 memory_filename。
        self.config: TreeTextMemoryConfig = config
        # 根据 extractor_llm 配置创建抽取模型；它主要用于记忆组织或 MemoryManager 内部结构整理。
        self.extractor_llm: OpenAILLM | OllamaLLM | AzureLLM = LLMFactory.from_config(
            # 传入抽取模型配置，由 LLMFactory 解析具体后端。
            config.extractor_llm
        # 结束当前多行调用或结构定义。
        )
        # 根据 dispatcher_llm 配置创建调度/检索模型；搜索器会用它做复杂检索流程中的理解和路由。
        self.dispatcher_llm: OpenAILLM | OllamaLLM | AzureLLM = LLMFactory.from_config(
            # 传入调度模型配置，由 LLMFactory 创建对应 LLM 实例。
            config.dispatcher_llm
        # 结束当前多行调用或结构定义。
        )
        # 根据 embedder 配置创建向量化组件；新增结构节点、embedding 检索和搜索器都依赖它。
        self.embedder: OllamaEmbedder = EmbedderFactory.from_config(config.embedder)
        # 根据 graph_db 配置创建图存储；所有节点和边的读写最终都落到这里。
        self.graph_store: Neo4jGraphDB = GraphStoreFactory.from_config(config.graph_db)

        # 保存搜索策略配置；后续决定是否启用 BM25、如何组合检索策略等。
        self.search_strategy = config.search_strategy
        # 根据搜索策略条件初始化 BM25 检索器；未启用时保持为 None。
        self.bm25_retriever = (
            # 只有配置存在且 bm25 开关为真时才创建 EnhancedBM25，否则搜索器不会走 BM25 召回。
            EnhancedBM25() if self.search_strategy and self.search_strategy["bm25"] else None
        # 结束当前多行调用或结构定义。
        )

        # 如果没有提供重排器配置，就构造一个默认的本地余弦重排配置。
        if config.reranker is None:
            # 用配置工厂验证并生成默认 reranker 配置对象，避免手写 dict 直接传入工厂。
            default_cfg = RerankerConfigFactory.model_validate(
                {
                    # 默认使用本地余弦相似度重排后端，不依赖额外在线模型。
                    "backend": "cosine_local",
                    # 开始定义默认重排器的内部参数。
                    "config": {
                        # 为不同层级背景设置相同权重，表示 topic/concept/fact 在默认重排中同等重要。
                        "level_weights": {"topic": 1.0, "concept": 1.0, "fact": 1.0},
                        # 指定从 metadata.background 字段读取层级或背景信息参与重排。
                        "level_field": "background",
                    # 结束当前字典配置片段。
                    },
                # 结束当前字典配置片段。
                }
            # 结束当前多行调用或结构定义。
            )
            # 根据默认配置创建 reranker 实例。
            self.reranker = RerankerFactory.from_config(default_cfg)
        # 如果提供演化目标，则同时记录 evolve_to。
        else:
            # 根据外部传入的 reranker 配置创建重排器。
            self.reranker = RerankerFactory.from_config(config.reranker)
        # 保存是否启用图结构重组；MemoryManager 初始化时会继续传入该开关。
        self.is_reorganize = config.reorganize
        # 创建 MemoryManager，把底层图写入、容量控制和图结构整理委托出去。
        self.memory_manager: MemoryManager = MemoryManager(
            # 传入图存储，搜索器从中召回节点和子图。
            self.graph_store,
            # 传入向量器，MemoryManager 在创建结构节点或重组图时可能需要向量化文本。
            self.embedder,
            # 传入抽取 LLM，供 MemoryManager 内部的图结构重组器使用。
            self.extractor_llm,
            # 优先使用配置中提供的记忆容量上限。
            memory_size=config.memory_size
            # 如果配置没有提供 memory_size，则回退到默认容量配置。
            or {
                # 默认最多保留 20 条工作记忆。
                "WorkingMemory": 20,
                # 默认长期记忆容量上限为 1500。
                "LongTermMemory": 1500,
                # 默认用户记忆容量上限为 480。
                "UserMemory": 480,
            # 结束当前字典配置片段。
            },
            # 把重组开关传入 MemoryManager，使新增节点后是否触发重组由配置控制。
            is_reorganize=self.is_reorganize,
        # 结束当前多行调用或结构定义。
        )
        # Create internet retriever if configured
        # 先把互联网检索器置空；只有配置显式开启时才会创建。
        self.internet_retriever = None
        # 如果配置提供互联网检索器参数，就创建对应检索后端。
        if config.internet_retriever is not None:
            # 通过工厂创建互联网检索器，并将 embedder 传入以支持联网内容的向量化或融合。
            self.internet_retriever = InternetRetrieverFactory.from_config(
                # 传入互联网检索配置和向量器。
                config.internet_retriever, self.embedder
            # 结束当前多行调用或结构定义。
            )
            # 记录关键初始化或运行状态，便于排查配置是否生效。
            logger.info(
                # 输出互联网检索器的实际后端名称。
                f"Internet retriever initialized with backend: {config.internet_retriever.backend}"
            # 结束当前多行调用或结构定义。
            )
        # 如果提供演化目标，则同时记录 evolve_to。
        else:
            # 记录未配置互联网检索器；搜索时不会自动联网扩展。
            logger.info("No internet retriever configured")
        # 预留 tokenizer 字段；搜索器初始化时会传入，便于未来支持自定义分词器。
        self.tokenizer = None
        # 保存是否在搜索或导出结果中包含 embedding；默认不包含以降低返回体积。
        self.include_embedding = config.include_embedding or False

    # 对外新增记忆入口；这里只做门面转发，真正写入逻辑由 MemoryManager 负责。
    def add(
        self,
        # 接收 TextualMemoryItem 或 dict 形式的记忆列表，兼容不同上游调用方式。
        memories: list[TextualMemoryItem | dict[str, Any]],
        # user_name 用于多租户或 cube 级隔离，传给底层图存储。
        user_name: str | None = None,
        # 透传额外搜索参数，保持接口扩展性。
        **kwargs,
    # 返回成功写入的记忆节点 ID 列表。
    ) -> list[str]:
        """Add memories.
        Args:
            memories: List of TextualMemoryItem objects or dictionaries to add.
            user_name: optional user_name
        """
        # 将写入请求委托给 MemoryManager，并把当前 TreeTextMemory 的 mode 传下去。
        return self.memory_manager.add(memories, user_name=user_name, mode=self.mode)

    # 替换 WorkingMemory 的门面方法；实际容量控制和写库由 MemoryManager 完成。
    def replace_working_memory(
        # 计算并保存 self, memories: list[TextualMemoryItem], user_name: str | None，供后续逻辑继续使用。
        self, memories: list[TextualMemoryItem], user_name: str | None = None
    ) -> None:
        # 把新的工作记忆列表交给 MemoryManager，用它刷新短期工作记忆窗口。
        self.memory_manager.replace_working_memory(memories, user_name=user_name)

    # 读取当前 WorkingMemory，并按更新时间从新到旧排序返回。
    def get_working_memory(self, user_name: str | None = None) -> list[TextualMemoryItem]:
        # 从图数据库中查询所有 WorkingMemory 原始记录。
        working_memories = self.graph_store.get_all_memory_items(
            # 限定查询范围为 WorkingMemory，并按用户或 cube 命名空间隔离。
            scope="WorkingMemory", user_name=user_name
        # 结束当前多行调用或结构定义。
        )
        # 把图数据库返回的 dict 记录转换成 TextualMemoryItem 对象。
        items = [TextualMemoryItem.from_dict(record) for record in (working_memories)]
        # Sort by updated_at in descending order
        # 对工作记忆进行排序，确保最新的记忆排在最前。
        sorted_items = sorted(
            # 使用 metadata.updated_at 作为排序键；缺失时间时使用最小时间兜底。
            items, key=lambda x: x.metadata.updated_at or datetime.min, reverse=True
        # 结束当前多行调用或结构定义。
        )
        # 返回排序后的 WorkingMemory 列表。
        return sorted_items

    # 获取当前各类记忆数量；这里继续委托给 MemoryManager。
    def get_current_memory_size(self, user_name: str | None = None) -> dict[str, int]:
        """
        Get the current size of each memory type.
        This delegates to the MemoryManager.
        """
        # 由 MemoryManager 刷新并返回按 memory_type 分组的数量。
        return self.memory_manager.get_current_memory_size(user_name=user_name)

    # 创建并返回一个 Searcher 实例，供外部复用当前 TreeTextMemory 的检索依赖。
    def get_searcher(
        # manual_close_internet 控制搜索器是否关闭联网；moscube 当前未使用；process_llm 可替换搜索过程中的处理模型。
        self, manual_close_internet: bool = False, moscube: bool = False, process_llm=None
    # 结束函数签名，下面进入方法主体。
    ):
        # 开始构造高级搜索器。
        searcher = Searcher(
            # 传入调度 LLM，搜索器会用它处理复杂检索理解或推理流程。
            self.dispatcher_llm,
            # 传入图存储，搜索器从中召回节点和子图。
            self.graph_store,
            # 传入向量器，MemoryManager 在创建结构节点或重组图时可能需要向量化文本。
            self.embedder,
            # 传入重排器，用于对召回结果重新排序。
            self.reranker,
            # 传入 BM25 检索器；如果未启用则为 None。
            bm25_retriever=self.bm25_retriever,
            # 该便捷方法默认不传互联网检索器，避免外部获取 searcher 时自动联网。
            internet_retriever=None,
            # 传入搜索策略，使 searcher 按配置选择召回和融合方式。
            search_strategy=self.search_strategy,
            # 把联网控制参数传给 searcher。
            manual_close_internet=manual_close_internet,
            # 把可选处理模型传入搜索器，用于定制搜索中的 LLM 处理环节。
            process_llm=process_llm,
            # 传入 tokenizer 预留字段。
            tokenizer=self.tokenizer,
            # 控制 searcher 返回结果中是否包含 embedding。
            include_embedding=self.include_embedding,
        # 结束当前多行调用或结构定义。
        )
        # 返回创建好的搜索器实例。
        return searcher

    # 对外搜索入口；负责根据参数临时构造 Searcher 并执行搜索。
    def search(
        self,
        # 用户查询文本，是记忆检索的核心输入。
        query: str,
        # 希望返回的最高相关结果数量。
        top_k: int,
        # 可选上下文信息，通常用于记录记忆消费或辅助搜索。
        info=None,
        # 搜索模式，fast 更快，fine 更精细。
        mode: str = "fast",
        # 限定搜索的记忆类型；All 表示不做类型限制。
        memory_type: str = "All",
        # 默认关闭联网检索，避免一次普通搜索意外触发互联网访问。
        manual_close_internet: bool = True,
        # 可选搜索优先级条件，通常用于 session 等维度优先召回。
        search_priority: dict | None = None,
        # 可选 metadata 过滤条件，用于限制返回结果范围。
        search_filter: dict | None = None,
        # user_name 用于多租户或 cube 级隔离，传给底层图存储。
        user_name: str | None = None,
        # 是否同时检索工具相关记忆。
        search_tool_memory: bool = False,
        # 工具记忆的召回数量上限。
        tool_mem_top_k: int = 6,
        # 是否包含技能记忆。
        include_skill_memory: bool = False,
        # 技能记忆的召回数量上限。
        skill_mem_top_k: int = 3,
        # 是否包含偏好记忆。
        include_preference_memory: bool = False,
        # 偏好记忆的召回数量上限。
        pref_mem_top_k: int = 6,
        # 可选去重模式，透传给搜索器。
        dedup: str | None = None,
        # 单次搜索是否返回 embedding；不传时使用实例默认设置。
        include_embedding: bool | None = None,
        # 透传额外搜索参数，保持接口扩展性。
        **kwargs,
    ) -> list[TextualMemoryItem]:
        """Search for memories based on a query.
        User query -> TaskGoalParser -> MemoryPathResolver ->
        GraphMemoryRetriever -> MemoryReranker -> MemoryReasoner -> Final output
        Args:
            query (str): The query to search for.
            top_k (int): The number of top results to return.
            info (dict): Leave a record of memory consumption.
            mode (str, optional): The mode of the search.
            - 'fast': Uses a faster search process, sacrificing some precision for speed.
            - 'fine': Uses a more detailed search process, invoking large models for higher precision, but slower performance.
            memory_type (str): Type restriction for search.
            ['All', 'WorkingMemory', 'LongTermMemory', 'UserMemory']
            manual_close_internet (bool): If True, the internet retriever will be closed by this search, it high priority than config.
            search_filter (dict, optional): Optional metadata filters for search results.
                - Keys correspond to memory metadata fields (e.g., "user_id", "session_id").
                - Values are exact-match conditions.
                Example: {"user_id": "123", "session_id": "abc"}
                If None, no additional filtering is applied.
        Returns:
            list[TextualMemoryItem]: List of matching memories.
        """
        # Use parameter if provided, otherwise fall back to instance attribute
        # 确定本次搜索是否包含 embedding，允许单次调用覆盖全局配置。
        include_emb = include_embedding if include_embedding is not None else self.include_embedding

        # 开始构造高级搜索器。
        searcher = Searcher(
            # 传入调度 LLM，搜索器会用它处理复杂检索理解或推理流程。
            self.dispatcher_llm,
            # 传入图存储，搜索器从中召回节点和子图。
            self.graph_store,
            # 传入向量器，MemoryManager 在创建结构节点或重组图时可能需要向量化文本。
            self.embedder,
            # 传入重排器，用于对召回结果重新排序。
            self.reranker,
            # 传入 BM25 检索器；如果未启用则为 None。
            bm25_retriever=self.bm25_retriever,
            # 在 search 方法中传入实例配置好的互联网检索器，使搜索器可以按参数决定是否联网。
            internet_retriever=self.internet_retriever,
            # 传入搜索策略，使 searcher 按配置选择召回和融合方式。
            search_strategy=self.search_strategy,
            # 把联网控制参数传给 searcher。
            manual_close_internet=manual_close_internet,
            # 传入 tokenizer 预留字段。
            tokenizer=self.tokenizer,
            # 使用本次解析后的 include_embedding 设置初始化搜索器。
            include_embedding=include_emb,
        # 结束当前多行调用或结构定义。
        )
        # 把所有检索参数透传给 Searcher.search，并返回其检索结果。
        return searcher.search(
            # 传入查询文本。
            query,
            # 传入主结果数量上限。
            top_k,
            # 传入搜索上下文信息。
            info,
            # 传入搜索模式。
            mode,
            # 传入记忆类型限制。
            memory_type,
            # 传入过滤条件。
            search_filter,
            # 传入优先级条件。
            search_priority,
            # 传入用户或 cube 命名空间，保证检索隔离。
            user_name=user_name,
            # 传入是否搜索工具记忆的开关。
            search_tool_memory=search_tool_memory,
            # 传入工具记忆召回数量。
            tool_mem_top_k=tool_mem_top_k,
            # 传入是否包含技能记忆。
            include_skill_memory=include_skill_memory,
            # 传入技能记忆召回数量。
            skill_mem_top_k=skill_mem_top_k,
            # 传入是否包含偏好记忆。
            include_preference_memory=include_preference_memory,
            # 传入偏好记忆召回数量。
            pref_mem_top_k=pref_mem_top_k,
            # 传入去重策略。
            dedup=dedup,
            # 透传额外搜索参数，保持接口扩展性。
            **kwargs,
        # 结束当前多行调用或结构定义。
        )

    # 根据查询召回相关节点，并合并这些节点周围的局部子图。
    def get_relevant_subgraph(
        self,
        # 用户查询文本，是记忆检索的核心输入。
        query: str,
        # 指定作为子图中心候选的相关节点数量。
        top_k: int = 20,
        # 指定从中心节点向外扩展的图跳数。
        depth: int = 2,
        # 只以指定状态的节点作为有效中心，默认只看 activated 节点。
        center_status: str = "activated",
        # user_name 用于多租户或 cube 级隔离，传给底层图存储。
        user_name: str | None = None,
        # 限定子图中心召回方式，支持向量检索和全文检索两种。
        search_type: Literal["embedding", "fulltext"] = "fulltext",
    ) -> dict[str, Any]:
        """
        Find and merge the local neighborhood sub-graphs of the top-k
        nodes most relevant to the query.
         Process:
             1. Embed the user query into a vector representation.
             2. Use vector similarity search to find the top-k similar nodes.
             3. For each similar node:
                 - Ensure its status matches `center_status` (e.g., 'active').
                 - Retrieve its local subgraph up to `depth` hops.
                 - Collect the center node, its neighbors, and connecting edges.
             4. Merge all retrieved subgraphs into a single unified subgraph.
             5. Return the merged subgraph structure.

         Args:
             query (str): The user input or concept to find relevant memories for.
             top_k (int, optional): How many top similar nodes to retrieve. Default is 5.
             depth (int, optional): The neighborhood depth (number of hops). Default is 2.
             center_status (str, optional): Status condition the center node must satisfy (e.g., 'active').

         Returns:
             dict[str, Any]: A subgraph dict with:
                 - 'core_id': ID of the top matching core node, or None if none found.
                 - 'nodes': List of unique nodes (core + neighbors) in the merged subgraph.
                 - 'edges': List of unique edges (as dicts with 'from', 'to', 'type') in the merged subgraph.
        """
        # 如果选择 embedding 模式，则先向量化查询再做向量相似度搜索。
        if search_type == "embedding":
            # Step 1: Embed query
            # 把用户查询转换成向量表示，取批量结果中的第一个向量。
            query_embedding = self.embedder.embed([query])[0]

            # Step 2: Get top-1 similar node
            # 在图数据库中按 embedding 相似度搜索相关节点。
            similar_nodes = self.graph_store.search_by_embedding(
                # 传入查询向量、返回数量和用户命名空间。
                query_embedding, top_k=top_k, user_name=user_name
            # 结束当前多行调用或结构定义。
            )

        # 如果选择全文检索，则按查询词匹配图节点文本。
        elif search_type == "fulltext":

            # 装饰内部函数，确保中文分词依赖 jieba 在使用前可用。
            @require_python_package(
                # 声明需要检查的 Python 包名。
                import_name="jieba",
                # 给出缺失依赖时的安装命令。
                install_command="pip install jieba",
                # 提供 jieba 项目链接，方便定位依赖来源。
                install_link="https://github.com/fxsjy/jieba",
            # 结束当前多行调用或结构定义。
            )
            # 定义中文分词函数，只在 fulltext 且查询为中文时使用。
            def _tokenize_chinese(text):
                """split zh jieba"""
                # 在函数内部延迟导入 jieba，避免未使用中文检索时强依赖该包。
                import jieba

                # 创建停用词管理器，用于过滤无意义词。
                stopword_manager = StopwordManager()
                # 使用 jieba 对中文文本进行分词。
                tokens = jieba.lcut(text)
                # 去掉空白 token，并过滤空字符串。
                tokens = [token.strip() for token in tokens if token.strip()]
                # 过滤停用词后返回最终查询词列表。
                return stopword_manager.filter_words(tokens)

            # 检测查询语言，用于决定是否需要中文分词。
            lang = detect_lang(query)
            # 中文查询使用 jieba 分词，非中文查询使用空格切分。
            queries = _tokenize_chinese(query) if lang == "zh" else query.split()

            # 在图数据库中按全文词项搜索相关节点。
            similar_nodes = self.graph_store.search_by_fulltext(
                # 传入分词后的查询词列表。
                query_words=queries,
                # 计算并保存 top_k，供后续逻辑继续使用。
                top_k=top_k,
                # 传入用户或 cube 命名空间，保证检索隔离。
                user_name=user_name,
            # 结束当前多行调用或结构定义。
            )

        # 如果没有召回任何相关节点，就直接返回空子图。
        if not similar_nodes:
            # 记录没有找到相关节点，虽然日志文案提到 embedding，但 fulltext 模式也会走到这里。
            logger.info("No similar nodes found for query embedding.")
            # 返回统一的空子图结构，避免调用方处理 None。
            return {"core_id": None, "nodes": [], "edges": []}

        # Step 3: Fetch neighborhood
        # 用字典聚合所有子图节点，以节点 ID 去重。
        all_nodes = {}
        # 用 set 聚合所有边，以 source、target、type 三元组去重。
        all_edges = set()
        # 保存每个中心节点及其分数和邻域信息，用于最后确定 top core。
        cores = []

        # 遍历每个召回的相关节点，将其邻域子图合并到总结果中。
        for node in similar_nodes:
            # 取出当前中心节点 ID。
            core_id = node["id"]
            # 取出当前中心节点的相关性分数。
            score = node["score"]

            # 从图数据库读取以当前节点为中心的局部子图。
            subgraph = self.graph_store.get_subgraph(
                # 传入中心节点、扩展深度、中心状态过滤和用户命名空间。
                center_id=core_id, depth=depth, center_status=center_status, user_name=user_name
            # 结束当前多行调用或结构定义。
            )

            # 如果图数据库没有返回有效子图或缺少中心节点，就尝试降级处理。
            if subgraph is None or not subgraph["core_node"]:
                # 直接读取中心节点本身，避免完全丢失召回结果。
                node = self.graph_store.get_node(core_id, user_name=user_name)
                # 把中心节点作为邻居放入结果，作为子图缺失时的兜底。
                subgraph["neighbors"] = [node]

            # 取出局部子图的中心节点。
            core_node = subgraph["core_node"]
            # 取出局部子图中的邻居节点。
            neighbors = subgraph["neighbors"]
            # 取出局部子图中的边。
            edges = subgraph["edges"]

            # Collect nodes
            # 如果中心节点有效，就加入全局节点集合。
            if core_node:
                # 按节点 ID 写入字典，实现节点去重。
                all_nodes[core_node["id"]] = core_node
            # 遍历邻居节点。
            for n in neighbors:
                # 将邻居节点加入全局节点集合，同 ID 会覆盖去重。
                all_nodes[n["id"]] = n

            # Collect edges
            # 遍历当前局部子图的边。
            for e in edges:
                # 用三元组保存边，实现跨子图的边去重。
                all_edges.add((e["source"], e["target"], e["type"]))

            # 保存当前中心节点的完整信息，后续用于返回 core_id。
            cores.append(
                # 记录中心节点 ID、分数、中心节点内容和邻居节点。
                {"id": core_id, "score": score, "core_node": core_node, "neighbors": neighbors}
            # 结束当前多行调用或结构定义。
            )

        # 选择第一个召回节点作为核心节点；这里默认 similar_nodes 已按相关性排序。
        top_core = cores[0] if cores else None
        # 返回字典结构结果，保持调用方可稳定读取字段。
        return {
            # 返回最相关中心节点 ID；如果没有中心则为 None。
            "core_id": top_core["id"] if top_core else None,
            # 返回合并并去重后的节点列表。
            "nodes": list(all_nodes.values()),
            # 把边集合转换回字典列表，保持对外返回结构清晰。
            "edges": [{"source": f, "target": t, "type": ty} for (f, t, ty) in all_edges],
        # 结束当前字典配置片段。
        }

    # 抽取接口在该类中尚未实现，通常由更上层 mem_reader 负责生成 TextualMemoryItem。
    def extract(self, messages: MessageList) -> list[TextualMemoryItem]:
        # 明确告诉调用方该方法未实现，避免静默返回错误结果。
        raise NotImplementedError

    # 更新接口在当前实现中尚未提供。
    def update(self, memory_id: str, new_memory: TextualMemoryItem | dict[str, Any]) -> None:
        # 明确告诉调用方该方法未实现，避免静默返回错误结果。
        raise NotImplementedError

    # 根据 memory_id 从图数据库读取单条记忆，并还原成 TextualMemoryItem。
    def get(self, memory_id: str, user_name: str | None = None) -> TextualMemoryItem:
        """Get a memory by its ID."""
        # 向图数据库查询指定 ID 的节点。
        result = self.graph_store.get_node(memory_id, user_name=user_name)
        # 如果没有找到节点，就记录详细诊断信息并抛错。
        if result is None:
            # 输出 warning 级别日志，辅助定位查询失败是否来自 user_name、数据库或配置问题。
            logger.warning(
                # 计算并保存 "[TreeTextMemory.get] Memory not found. memory_id，供后续逻辑继续使用。
                "[TreeTextMemory.get] Memory not found. memory_id=%s, lookup_user_name=%s, graph_store=%s, db_name=%s, config_user_name=%s",
                # 日志中包含要查找的 memory_id。
                memory_id,
                user_name,
                type(self.graph_store).__name__,
                # 日志中包含图数据库名称，帮助判断是否查错库。
                getattr(self.graph_store, "db_name", None),
                # 日志中包含 graph_store 配置中的默认 user_name，帮助排查命名空间不一致。
                getattr(getattr(self.graph_store, "config", None), "user_name", None),
            # 结束当前多行调用或结构定义。
            )
            # 以 ValueError 告知调用方指定记忆不存在。
            raise ValueError(f"Memory with ID {memory_id} not found")
        # 从图节点结果中取出 metadata；缺失时使用空字典兜底。
        metadata_dict = result.get("metadata", {})
        # 构造并返回标准 TextualMemoryItem，让外部拿到统一的记忆对象。
        return TextualMemoryItem(
            # 用图节点 ID 设置 TextualMemoryItem.id。
            id=result["id"],
            # 用图节点 memory 字段设置记忆正文。
            memory=result["memory"],
            # 把 metadata 字典还原为 TreeNodeTextualMemoryMetadata 对象。
            metadata=TreeNodeTextualMemoryMetadata(**metadata_dict),
        # 结束当前多行调用或结构定义。
        )

    # 批量按 ID 获取节点。
    def get_by_ids(
        # 计算并保存 self, memory_ids: list[str], user_name: str | None，供后续逻辑继续使用。
        self, memory_ids: list[str], user_name: str | None = None
    ) -> list[TextualMemoryItem]:
        # 调用图数据库批量读取接口，按用户命名空间隔离。
        graph_output = self.graph_store.get_nodes(ids=memory_ids, user_name=user_name)
        # 直接返回图数据库输出；这里不额外转换成 TextualMemoryItem。
        return graph_output

    # 导出或分页获取当前图中的记忆。
    def get_all(
        self,
        # user_name 用于多租户或 cube 级隔离，传给底层图存储。
        user_name: str | None = None,
        # 用户 ID，仅用于日志记录。
        user_id: str | None = None,
        # 可选页码，用于分页导出。
        page: int | None = None,
        # 可选每页数量。
        page_size: int | None = None,
        # 可选通用过滤条件。
        filter: dict | None = None,
        # 可选记忆类型列表，用于只导出指定类型。
        memory_type: list[str] | None = None,
    ) -> dict:
        """Get all memories.
        Returns:
            list[TextualMemoryItem]: List of all memories.
        """
        # 调用图数据库导出接口，返回节点和边等图结构数据。
        graph_output = self.graph_store.export_graph(
            # 传入用户或 cube 命名空间，保证检索隔离。
            user_name=user_name,
            # 传入 user_id 过滤条件，区别于 user_name 命名空间。
            user_id=user_id,
            # 传入分页页码。
            page=page,
            # 传入分页大小。
            page_size=page_size,
            # 传入 metadata 或其他过滤条件。
            filter=filter,
            # 传入记忆类型过滤列表。
            memory_type=memory_type,
        # 结束当前多行调用或结构定义。
        )
        # 直接返回图数据库输出；这里不额外转换成 TextualMemoryItem。
        return graph_output

    # 硬删除指定记忆节点及其边。
    def delete(self, memory_ids: list[str], user_name: str | None = None) -> None:
        """Hard delete: permanently remove nodes and their edges from the graph."""
        # 如果没有传入 ID，直接返回，避免无意义数据库调用。
        if not memory_ids:
            # 显式返回 None，表示软删除流程结束。
            return
        # 遍历所有要软删除的记忆 ID。
        for mid in memory_ids:
            # 进入受保护执行块，下面的文件、数据库或并发操作都可能失败。
            try:
                # 调用图数据库删除节点；底层应同时移除相关边。
                self.graph_store.delete_node(mid, user_name=user_name)
            # 捕获单个节点删除失败。
            except Exception as e:
                # 记录失败节点 ID 和异常原因。
                logger.warning(f"TreeTextMemory.delete_hard: failed to delete {mid}: {e}")

    # 通过 memory_ids 参数调用图存储的条件删除接口。
    def delete_by_memory_ids(self, memory_ids: list[str]) -> None:
        """Delete memories by memory_ids."""
        # 进入受保护执行块，下面的文件、数据库或并发操作都可能失败。
        try:
            # 把 memory_ids 传给底层删除方法。
            self.graph_store.delete_node_by_prams(memory_ids=memory_ids)
        # 捕获单个节点删除失败。
        except Exception as e:
            # 记录按 ID 条件删除失败的异常。
            logger.error(f"An error occurred while deleting memories by memory_ids: {e}")

    # 删除指定命名空间下的全部记忆和关系。
    def delete_all(self, user_name: str | None = None) -> None:
        """Delete all memories and their relationships from the graph store."""
        # 进入受保护执行块，下面的文件、数据库或并发操作都可能失败。
        try:
            # 清空图数据库中该 user_name 下的节点和边。
            self.graph_store.clear(user_name=user_name)
            # 记录清空成功。
            logger.info("All memories and edges have been deleted from the graph.")
        # 捕获单个节点删除失败。
        except Exception as e:
            # 记录清空失败。
            logger.error(f"An error occurred while deleting all memories: {e}")
            # 继续向上抛出异常，让调用方感知 delete_all 失败。
            raise

    # 按 writable_cube_ids、file_ids 或 filter 删除节点。
    def delete_by_filter(
        self,
        # 可选可写 cube ID 列表，用于按 cube 范围删除。
        writable_cube_ids: list[str] | None = None,
        # 可选文件 ID 列表，用于删除指定文件相关记忆。
        file_ids: list[str] | None = None,
        # 可选通用过滤条件。
        filter: dict | None = None,
    ) -> None:
        """Delete memories by filter."""
        # 调用图数据库的参数化删除接口。
        self.graph_store.delete_node_by_prams(
            # 把过滤条件直接传给底层存储执行。
            writable_cube_ids=writable_cube_ids, file_ids=file_ids, filter=filter
        # 结束当前多行调用或结构定义。
        )

    # 从目录中的 JSON 文件加载记忆图数据到图数据库。
    def load(self, dir: str, user_name: str | None = None) -> None:
        # 进入受保护执行块，下面的文件、数据库或并发操作都可能失败。
        try:
            # 拼出导出文件路径。
            memory_file = os.path.join(dir, self.config.memory_filename)

            # 如果文件不存在，记录 warning 后返回。
            if not os.path.exists(memory_file):
                # 提示指定记忆文件不存在。
                logger.warning(f"Memory file not found: {memory_file}")
                # 显式返回 None，表示软删除流程结束。
                return

            # 以 UTF-8 打开记忆 JSON 文件。
            with open(memory_file, encoding="utf-8") as f:
                # 解析 JSON 内容，得到待导入的图数据。
                memories = json.load(f)

            # 把 JSON 图数据导入图数据库。
            self.graph_store.import_graph(memories, user_name=user_name)
            # 记录导入数量和文件路径。
            logger.info(f"Loaded {len(memories)} memories from {memory_file}")

        # 捕获文件不存在异常，虽然前面已检查，但这里作为兜底。
        except FileNotFoundError:
            # 记录目录中找不到记忆文件。
            logger.error(f"Memory file not found in directory: {dir}")
        # 捕获 JSON 格式错误。
        except json.JSONDecodeError as e:
            # 记录 JSON 解码失败原因。
            logger.error(f"Error decoding JSON from memory file: {e}")
        # 捕获单个节点删除失败。
        except Exception as e:
            # 记录加载过程中的其他异常。
            logger.error(f"An error occurred while loading memories: {e}")

    # 把图数据库中的记忆导出到指定目录。
    def dump(self, dir: str, include_embedding: bool = False, user_name: str | None = None) -> None:
        """Dump memories to os.path.join(dir, self.config.memory_filename)"""
        # 进入受保护执行块，下面的文件、数据库或并发操作都可能失败。
        try:
            # 从图数据库导出图结构数据。
            json_memories = self.graph_store.export_graph(
                # 控制是否导出 embedding，并限定用户命名空间。
                include_embedding=include_embedding, user_name=user_name
            # 结束当前多行调用或结构定义。
            )

            # 确保目标目录存在；已存在时不报错。
            os.makedirs(dir, exist_ok=True)
            # 拼出导出文件路径。
            memory_file = os.path.join(dir, self.config.memory_filename)
            # 以 UTF-8 写模式打开目标文件。
            with open(memory_file, "w", encoding="utf-8") as f:
                # 把图数据格式化写入 JSON，保留中文字符不转义。
                json.dump(json_memories, f, indent=4, ensure_ascii=False)

            # 记录导出的节点数量和文件路径。
            logger.info(f"Dumped {len(json_memories.get('nodes'))} memories to {memory_file}")

        # 捕获单个节点删除失败。
        except Exception as e:
            # 记录导出失败。
            logger.error(f"An error occurred while dumping memories: {e}")
            # 继续向上抛出异常，让调用方感知 delete_all 失败。
            raise

    # 备份当前记忆数据后删除整个 Neo4j 数据库。
    def drop(self, keep_last_n: int = 30) -> None:
        """
        Export all memory data to a versioned backup dir and drop the Neo4j database.
        Only the latest `keep_last_n` backups will be retained.
        """
        # 进入受保护执行块，下面的文件、数据库或并发操作都可能失败。
        try:
            # 在系统临时目录下创建统一备份根目录。
            backup_root = Path(tempfile.gettempdir()) / "memos_backups"
            # 确保备份根目录存在，必要时递归创建。
            backup_root.mkdir(parents=True, exist_ok=True)

            # 生成时间戳，用于创建唯一备份目录。
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # 拼出本次备份目录路径。
            backup_dir = backup_root / f"memos_backup_{timestamp}"
            # 创建本次备份目录。
            backup_dir.mkdir()

            # 记录即将备份到哪个目录。
            logger.info(f"Exporting memory to backup dir: {backup_dir}")
            # 先导出当前图数据，确保 drop 前有备份。
            self.dump(str(backup_dir))

            # Clean up old backups
            # 清理旧备份，只保留最近 keep_last_n 个。
            self._cleanup_old_backups(backup_root, keep_last_n)

            # 调用图数据库接口删除整个数据库。
            self.graph_store.drop_database()
            # 记录数据库已在备份后被删除。
            logger.info(f"Database '{self.graph_store.db_name}' dropped after backup.")

        # 捕获单个节点删除失败。
        except Exception as e:
            # 记录 drop 过程失败。
            logger.error(f"Error in drop(): {e}")
            # 继续向上抛出异常，让调用方感知 delete_all 失败。
            raise

    # 定义静态方法；清理旧备份不依赖实例状态。
    @staticmethod
    # 清理备份根目录下超出保留数量的旧备份目录。
    def _cleanup_old_backups(root_dir: Path, keep_last_n: int) -> None:
        """
        Keep only the latest `keep_last_n` backup directories under `root_dir`.
        Older ones will be deleted.
        """
        # 列出并排序所有备份目录。
        backups = sorted(
            [d for d in root_dir.iterdir() if d.is_dir() and d.name.startswith("memos_backup_")],
            # 按目录名排序；目录名包含时间戳，因此可代表备份时间。
            key=lambda p: p.name,  # name includes timestamp
            # 倒序排列，让最新备份排在前面。
            reverse=True,
        # 结束当前多行调用或结构定义。
        )

        # 跳过需要保留的最新 N 个，其余作为待删除目录。
        to_delete = backups[keep_last_n:]
        # 逐个删除过期备份目录。
        for old_dir in to_delete:
            # 进入受保护执行块，下面的文件、数据库或并发操作都可能失败。
            try:
                # 递归删除整个备份目录。
                shutil.rmtree(old_dir)
                # 记录旧备份删除成功。
                logger.info(f"Deleted old backup directory: {old_dir}")
            # 捕获单个节点删除失败。
            except Exception as e:
                # 记录某个旧备份删除失败，但不影响其他备份清理。
                logger.warning(f"Failed to delete backup {old_dir}: {e}")

    # 写入 RawFileMemory 节点，并为原始文件块和摘要记忆建立图边。
    def add_rawfile_nodes_n_edges(
        self,
        # 待写入的原始文件块记忆列表。
        raw_file_mem_group: list[TextualMemoryItem],
        # 已经写入的摘要/普通记忆节点 ID 列表，用于建立 SUMMARY/MATERIAL 边。
        mem_ids: list[str],
        # 用户 ID，仅用于日志记录。
        user_id: str | None = None,
        # user_name 用于多租户或 cube 级隔离，传给底层图存储。
        user_name: str | None = None,
    ) -> None:
        """
        Add raw file nodes and edges to the graph. Edges are between raw file ids and mem_ids.
        Args:
            raw_file_mem_group: List of raw file memory items.
            mem_ids: List of memory IDs.
            user_name: cube id.
        """
        # 先把 RawFileMemory 节点写入图数据库，并拿到实际写入的 rawfile ID。
        rawfile_ids_local: list[str] = self.add(
            # 传入原始文件块记忆列表。
            raw_file_mem_group,
            # 传入用户或 cube 命名空间，保证检索隔离。
            user_name=user_name,
        # 结束当前多行调用或结构定义。
        )

        # 准备批量添加边的起点 ID 列表。
        from_ids = []
        # 准备批量添加边的终点 ID 列表。
        to_ids = []
        # 准备批量添加边的类型列表，与 from_ids/to_ids 按位置一一对应。
        types = []

        # 遍历每个原始文件块，基于其 metadata 构建边。
        for raw_file_mem in raw_file_mem_group:
            # Add SUMMARY edge: memory -> raw file; raw file -> memory
            # 如果该原始文件块记录了 summary_ids，就建立它与摘要记忆之间的关系。
            if hasattr(raw_file_mem.metadata, "summary_ids") and raw_file_mem.metadata.summary_ids:
                # 读取该原始文件块对应的摘要节点 ID 列表。
                summary_ids = raw_file_mem.metadata.summary_ids
                # 遍历每个摘要节点 ID。
                for summary_id in summary_ids:
                    # 只为本次已写入的摘要节点建立边，避免引用不存在或不属于本批次的节点。
                    if summary_id in mem_ids:
                        # 添加摘要记忆到原始文件块的边起点。
                        from_ids.append(summary_id)
                        # 添加摘要记忆到原始文件块的边终点。
                        to_ids.append(raw_file_mem.id)
                        # MATERIAL 表示该摘要记忆来源于该原始材料块。
                        types.append("MATERIAL")

                        # 添加原始文件块到摘要记忆的反向边起点。
                        from_ids.append(raw_file_mem.id)
                        # 添加原始文件块到摘要记忆的反向边终点。
                        to_ids.append(summary_id)
                        # SUMMARY 表示该原始文件块对应某个摘要记忆。
                        types.append("SUMMARY")

            # Add FOLLOWING edge: current chunk -> next chunk
            # 开始判断是否需要添加文件块之间的顺序边。
            if (
                # 检查 metadata 中是否存在 following_id。
                hasattr(raw_file_mem.metadata, "following_id")
                # 确保 following_id 有实际值。
                and raw_file_mem.metadata.following_id
            # 结束函数签名，下面进入方法主体。
            ):
                # 读取下一个文件块 ID。
                following_id = raw_file_mem.metadata.following_id
                # 只有下一个文件块也在本次写入成功的 rawfile IDs 中，才建立 FOLLOWING 边。
                if following_id in rawfile_ids_local:
                    # 添加原始文件块到摘要记忆的反向边起点。
                    from_ids.append(raw_file_mem.id)
                    to_ids.append(following_id)
                    # FOLLOWING 表示当前文件块指向下一个文件块。
                    types.append("FOLLOWING")

            # Add PRECEDING edge: previous chunk -> current chunk
            # 开始判断是否需要添加文件块之间的顺序边。
            if (
                # 检查 metadata 中是否存在 preceding_id。
                hasattr(raw_file_mem.metadata, "preceding_id")
                # 确保 preceding_id 有实际值。
                and raw_file_mem.metadata.preceding_id
            # 结束函数签名，下面进入方法主体。
            ):
                # 读取上一个文件块 ID。
                preceding_id = raw_file_mem.metadata.preceding_id
                # 只有上一个文件块也在本次写入成功的 rawfile IDs 中，才建立 PRECEDING 边。
                if preceding_id in rawfile_ids_local:
                    # 添加原始文件块到摘要记忆的反向边起点。
                    from_ids.append(raw_file_mem.id)
                    to_ids.append(preceding_id)
                    # PRECEDING 表示当前文件块指向前一个文件块。
                    types.append("PRECEDING")

        # 记录批量添加边开始时间，用于计算耗时。
        start_time = time.time()
        # 调用统一的并发添加边方法。
        self.add_graph_edges(
            # 传入所有边的起点 ID。
            from_ids,
            # 传入所有边的终点 ID。
            to_ids,
            # 传入所有边的类型。
            types,
            # 传入用户或 cube 命名空间，保证检索隔离。
            user_name=user_name,
        # 结束当前多行调用或结构定义。
        )
        # 记录添加边结束时间。
        end_time = time.time()
        # 记录本次写入的 RawFile chunk 数量。
        logger.info(f"[RawFile] Added {len(rawfile_ids_local)} chunks for user {user_id}")
        # 记录关键初始化或运行状态，便于排查配置是否生效。
        logger.info(
            # 记录 RawFile 关系边写入耗时和边数量。
            f"[RawFile] Time taken to add edges: {end_time - start_time} seconds for {len(from_ids)} edges"
        # 结束当前多行调用或结构定义。
        )

    # 并发批量添加图边。
    def add_graph_edges(
        # 三个列表按位置对应，组成 from、to、edge_type 三元组。
        self, from_ids: list[str], to_ids: list[str], types: list[str], user_name: str | None = None
    ) -> None:
        """
        Add edges to the graph.
        Args:
            from_ids: List of source node IDs.
            to_ids: List of target node IDs.
            types: List of edge types.
            user_name: Optional user name.
        """
        # 创建最多 20 个线程并发写边，提高大量文件块关系写入速度。
        with ContextThreadPoolExecutor(max_workers=20) as executor:
            # 构造 Future 集合，每条边对应一个 add_edge 任务。
            futures = {
                # 提交单条边写入任务到线程池。
                executor.submit(
                    # 调用图数据库添加边接口，并传入用户命名空间。
                    self.graph_store.add_edge, from_id, to_id, edge_type, user_name=user_name
                # 结束当前多行调用或结构定义。
                )
                # 按位置同时遍历 from_ids、to_ids、types；长度不一致时 strict=False 会以最短列表为准。
                for from_id, to_id, edge_type in zip(from_ids, to_ids, types, strict=False)
            # 结束当前字典配置片段。
            }

            # 按完成顺序等待每条边写入任务。
            for future in concurrent.futures.as_completed(futures):
                # 进入受保护执行块，下面的文件、数据库或并发操作都可能失败。
                try:
                    # 取出任务结果；如果写边失败，这里会抛出异常。
                    future.result()
                # 捕获单个节点删除失败。
                except Exception as e:
                    # 记录边写入异常，但不中断其他边处理。
                    logger.exception("Add edge error: ", exc_info=e)

    # 软删除记忆：不物理删除节点，而是更新状态字段。
    def soft_delete(
        self,
        # 需要标记删除的记忆 ID 列表。
        memory_ids: list[str],
        # 用户或 cube 命名空间，确保只更新对应范围内的节点。
        user_name: str,
        # 可选演化目标 ID；用于记录这些被删记忆演化到了哪些新记忆。
        evolve_to_ids: list[str] | None = None,
    ) -> None:
        # for ruff check...
        # 如果没有演化目标，只标记为 deleted。
        if not evolve_to_ids:
            # 构造软删除字段，只更新状态。
            update_fields = {"status": "deleted"}
        # 如果提供演化目标，则同时记录 evolve_to。
        else:
            # 构造包含删除状态和演化目标的更新字段。
            update_fields = {"status": "deleted", "evolve_to": evolve_to_ids}

        # Execute the actual marking operation - in db.
        # 用线程池并发更新多个节点，提高批量软删除速度。
        with ContextThreadPoolExecutor() as executor:
            # 保存每个 update_node 任务的 Future。
            futures = []
            # 遍历所有要软删除的记忆 ID。
            for mid in memory_ids:
                # 把当前节点更新任务加入 Future 列表。
                futures.append(
                    # 提交单条边写入任务到线程池。
                    executor.submit(
                        # 调用图数据库节点更新接口。
                        self.graph_store.update_node,
                        # 指定要更新的节点 ID。
                        id=mid,
                        # 传入要更新的状态字段。
                        fields=update_fields,
                        # 传入用户或 cube 命名空间，保证检索隔离。
                        user_name=user_name,
                    # 结束当前多行调用或结构定义。
                    )
                # 结束当前多行调用或结构定义。
                )

            # Wait for all tasks to complete and raise any exceptions
            # 逐个等待所有更新任务完成。
            for future in futures:
                # 取出任务结果；如果写边失败，这里会抛出异常。
                future.result()
        # 显式返回 None，表示软删除流程结束。
        return
