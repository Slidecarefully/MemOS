# 本文件是 SimpleStructMemReader 的实现：负责把 chat/doc 等输入统一转成可写入记忆系统的 TextualMemoryItem。
# 注释按代码执行顺序补充，尽量解释每一步在“记忆抽取、过滤、构建、并发处理”链路中的作用。
# 导入 futures 并发工具，用于等待线程池中的多个记忆抽取任务完成。
import concurrent.futures
# 导入 copy，用于在过滤或窗口重叠时复制列表/对象，避免直接污染原始输入。
import copy
# 导入 json，用于序列化 prompt 输入、日志内容以及解析 LLM 返回结果。
import json
# 导入 os，用于读取环境变量开关，例如是否启用新增记忆过滤。
import os
# 导入 traceback，用于在并发任务失败时输出完整异常堆栈。
import traceback

# 导入 ABC，使当前 reader 保持抽象基类语义，便于和 BaseMemReader 体系对齐。
from abc import ABC
# 导入类型标注工具；TYPE_CHECKING 只在静态检查阶段引入重依赖，避免运行期循环导入。
from typing import TYPE_CHECKING, Any, TypeAlias

# 导入 tqdm，用于文档分块并发处理时显示处理进度。
from tqdm import tqdm

# 导入项目日志模块，统一使用 memos 内部 logger。
from memos import log
# 导入 ChunkerFactory，根据配置创建文档切分器。
from memos.chunkers import ChunkerFactory
# 导入 LLMConfigFactory，用于把 dict 形式的 LLM 配置校验并转成配置对象。
from memos.configs.llm import LLMConfigFactory
# 导入当前 reader 的配置模型，构造函数会依赖它初始化 LLM、embedder 和 chunker。
from memos.configs.mem_reader import SimpleStructMemReaderConfig
# 导入带上下文传播能力的线程池，保证并发任务里仍能拿到调用链上下文。
from memos.context.context import ContextThreadPoolExecutor
# 导入 EmbedderFactory，根据配置创建向量化组件。
from memos.embedders.factory import EmbedderFactory
# 导入 LLMFactory，根据配置创建主 LLM、通用 LLM 或偏好抽取 LLM。
from memos.llms.factory import LLMFactory
# 导入 BaseMemReader，当前类继承它来接入统一 mem_reader 接口。
from memos.mem_reader.base import BaseMemReader


# 这些导入只服务类型检查，不会在运行期执行，从而减少循环依赖和启动成本。
if TYPE_CHECKING:
    # 从项目或标准库模块导入当前文件需要的依赖。
    from memos.graph_dbs.base import BaseGraphDB
    # 从项目或标准库模块导入当前文件需要的依赖。
    from memos.memories.textual.tree_text_memory.retrieve.searcher import Searcher
    # 从项目或标准库模块导入当前文件需要的依赖。
    from memos.types.general_types import UserContext
# 导入输入标准化和语言检测工具：前者兼容旧/新 scene_data，后者选择中英文 prompt。
from memos.mem_reader.read_multi_modal import coerce_scene_data, detect_lang
# 从对应模块批量导入多个依赖，括号内逐项列出具体对象。
from memos.mem_reader.utils import (
    # 导入 token 估算函数，用于 chat 窗口切分。
    count_tokens_text,
    # 导入 key 派生函数，在 LLM 未给 key 时自动生成。
    derive_key,
    # 导入 JSON 解析器，用于解析 LLM 抽取结果。
    parse_json_result,
    # 导入 keep/filter 解析器，用于解析幻觉过滤或保留判断结果。
    parse_keep_filter_response,
    # 导入改写结果解析器，用于解析 rewrite_memories 的返回。
    parse_rewritten_response,
)
# 从对应模块批量导入多个依赖，括号内逐项列出具体对象。
from memos.memories.textual.item import (
    # 导入来源消息模型，用于记录记忆来自哪条 chat 或哪个文档。
    SourceMessage,
    # 导入文本记忆实体模型，是 reader 最终输出的核心对象。
    TextualMemoryItem,
    # 导入文本记忆 metadata 模型，保存身份、状态、标签、embedding 等信息。
    TreeNodeTextualMemoryMetadata,
)
# 从对应模块批量导入多个依赖，括号内逐项列出具体对象。
from memos.templates.mem_reader_prompts import (
    # 导入英文自定义标签提示词。
    CUSTOM_TAGS_INSTRUCTION,
    # 导入中文自定义标签提示词。
    CUSTOM_TAGS_INSTRUCTION_ZH,
    # 导入英文通用字符串读取提示词。
    GENERAL_STRUCT_STRING_READER_PROMPT,
    # 导入中文通用字符串读取提示词。
    GENERAL_STRUCT_STRING_READER_PROMPT_ZH,
    # 导入通用 prompt 映射，供改写和幻觉过滤使用。
    PROMPT_MAPPING,
    # 导入英文文档记忆抽取提示词。
    SIMPLE_STRUCT_DOC_READER_PROMPT,
    # 导入中文文档记忆抽取提示词。
    SIMPLE_STRUCT_DOC_READER_PROMPT_ZH,
    # 导入英文 chat 抽取示例。
    SIMPLE_STRUCT_MEM_READER_EXAMPLE,
    # 导入中文 chat 抽取示例。
    SIMPLE_STRUCT_MEM_READER_EXAMPLE_ZH,
    # 导入英文 chat 记忆抽取提示词。
    SIMPLE_STRUCT_MEM_READER_PROMPT,
    # 导入中文 chat 记忆抽取提示词。
    SIMPLE_STRUCT_MEM_READER_PROMPT_ZH,
)
# 导入 MessagesType，表示标准化后的单个对话/文档场景消息结构。
from memos.types import MessagesType
# 从对应模块批量导入多个依赖，括号内逐项列出具体对象。
from memos.types.openai_chat_completion_types import (
    # 导入 assistant 消息类型。
    ChatCompletionAssistantMessageParam,
    # 导入文本内容片段类型。
    ChatCompletionContentPartTextParam,
    # 导入 system 消息类型。
    ChatCompletionSystemMessageParam,
    # 导入 tool 消息类型。
    ChatCompletionToolMessageParam,
    # 导入 user 消息类型。
    ChatCompletionUserMessageParam,
    # 导入文件内容类型。
    File,
)
# 导入 timed 装饰器，用于统计关键方法的耗时。
from memos.utils import timed


# 定义一个占位 ParserFactory，主要满足测试套件或旧接口对该名称的依赖。
class ParserFactory:
    """Placeholder required by test suite."""

    # 这里不需要访问实例或类状态，因此使用静态方法。
    @staticmethod
    # 保留 from_config 接口形态，让调用方即使传入配置也能安全返回占位解析器。
    def from_config(_config):
        # 返回 None 表示当前分支没有可用结果或选择安全降级。
        return None


# 汇总允许作为 chat message 的 OpenAI 消息类型，后续可用于类型判断或兼容层。
ChatMessageClasses = (
    # 导入 system 消息类型。
    ChatCompletionSystemMessageParam,
    # 导入 user 消息类型。
    ChatCompletionUserMessageParam,
    # 导入 assistant 消息类型。
    ChatCompletionAssistantMessageParam,
    # 导入 tool 消息类型。
    ChatCompletionToolMessageParam,
)

# 定义原始内容片段类型，覆盖纯文本片段和文件对象两类输入。
RawContentClasses = (ChatCompletionContentPartTextParam, File)
# 给旧版 dict 消息定义别名；虽然已废弃，但保留可以兼容历史调用。
MessageDict: TypeAlias = dict[str, Any]  # (Deprecated) not supported in the future
# 定义 get_memory 可接受的场景输入联合类型，同时覆盖旧版 chat/doc 和新版 MessagesType。
SceneDataInput: TypeAlias = (
    list[list[MessageDict]]  # (Deprecated) legacy chat example: scenes -> messages
    | list[str]  # (Deprecated) legacy doc example: list of paths / pure text
    | list[MessagesType]  # new: list of scenes (each scene is MessagesType)
)


# 创建模块级 logger，后续所有 helper 和类方法共享这一个日志入口。
logger = log.get_logger(__name__)
# 建立 prompt 映射表，把业务类型和语言映射到对应模板，避免每次调用时到处写条件分支。
PROMPT_DICT = {
    # chat 类型 prompt 用于从对话里抽取结构化记忆。
    "chat": {
        "en": SIMPLE_STRUCT_MEM_READER_PROMPT,
        "zh": SIMPLE_STRUCT_MEM_READER_PROMPT_ZH,
        "en_example": SIMPLE_STRUCT_MEM_READER_EXAMPLE,
        "zh_example": SIMPLE_STRUCT_MEM_READER_EXAMPLE_ZH,
    },
    # doc 类型 prompt 用于从文档分块里抽取结构化记忆。
    "doc": {"en": SIMPLE_STRUCT_DOC_READER_PROMPT, "zh": SIMPLE_STRUCT_DOC_READER_PROMPT_ZH},
    # general_string 类型保留给通用字符串抽取场景。
    "general_string": {
        "en": GENERAL_STRUCT_STRING_READER_PROMPT,
        "zh": GENERAL_STRUCT_STRING_READER_PROMPT_ZH,
    },
    # custom_tags prompt 用于把用户传入的标签约束注入抽取提示词。
    "custom_tags": {"en": CUSTOM_TAGS_INSTRUCTION, "zh": CUSTOM_TAGS_INSTRUCTION_ZH},
}


# 这个顶层 helper 面向文档分块：对单个 chunk 调 LLM、解析 JSON、补 embedding，并构造 TextualMemoryItem。
def _build_node(idx, message, info, source_info, llm, parse_json_result, embedder):
    # generate
    # 将单个 chunk 的处理包在 try 中，保证某个 chunk 失败不会影响其他 chunk。
    try:
        # 调用传入的 LLM 生成原始文本结果，这里期待结果能被解析成 JSON。
        raw = llm.generate(message)
        # 如果 LLM 返回空内容，说明当前 chunk 没有可用抽取结果，需要放弃该节点。
        if not raw:
            # 记录空生成的输入，方便排查 prompt、模型或内容为空的问题。
            logger.warning(f"[LLM] Empty generation for input: {message}")
            # 返回 None 表示当前分支没有可用结果或选择安全降级。
            return None
    # 捕获 LLM 调用异常，避免文档并发处理时单个任务拖垮整体。
    except Exception as e:
        # 记录生成阶段异常，定位模型服务或网络调用问题。
        logger.error(f"[LLM] Exception during generation: {e}")
        # 返回 None 表示当前分支没有可用结果或选择安全降级。
        return None

    # parse_json_result
    # 将单个 chunk 的处理包在 try 中，保证某个 chunk 失败不会影响其他 chunk。
    try:
        # 解析 LLM 输出，通常期望包含 value、tags、key 等字段。
        chunk_res = parse_json_result(raw)
        # 如果解析结果为空，说明模型输出不符合预期或没有有效记忆。
        if not chunk_res:
            # 记录无法解析的原始输出，便于优化 prompt 或解析器。
            logger.warning(f"[Parse] Failed to parse result: {raw}")
            # 返回 None 表示当前分支没有可用结果或选择安全降级。
            return None
    # 捕获 LLM 调用异常，避免文档并发处理时单个任务拖垮整体。
    except Exception as e:
        # 记录 JSON 解析异常，和空解析结果区分开来。
        logger.error(f"[Parse] Exception during JSON parsing: {e}")
        # 返回 None 表示当前分支没有可用结果或选择安全降级。
        return None

    # 将单个 chunk 的处理包在 try 中，保证某个 chunk 失败不会影响其他 chunk。
    try:
        # 取出真正要写入记忆系统的文本，并去掉首尾空白。
        value = chunk_res.get("value", "").strip()
        # 没有 value 就无法形成有效记忆节点，因此直接跳过。
        if not value:
            # 记录空 value，说明 LLM 虽然返回了结构但核心内容缺失。
            logger.warning("[BuildNode] value is empty")
            # 返回 None 表示当前分支没有可用结果或选择安全降级。
            return None

        # 读取模型给出的标签，用于后续检索或分类。
        tags = chunk_res.get("tags", [])
        # 标签必须是列表；如果模型返回了字符串或其他类型，需要纠正。
        if not isinstance(tags, list):
            # 类型不合法时丢弃标签，优先保证 memory item 的结构稳定。
            tags = []

        # 读取模型生成的 key，作为该记忆的摘要键或索引辅助信息。
        key = chunk_res.get("key", None)

        # 对记忆文本生成 embedding，方便后续向量检索。
        embedding = embedder.embed([value])[0]

        # 复制 info，避免 pop user_id/session_id 时修改调用方传入的原始字典。
        info_ = info.copy()
        # 从 info 中取出 user_id，放入 metadata 顶层字段。
        user_id = info_.pop("user_id", "")
        # 从 info 中取出 session_id，放入 metadata 顶层字段。
        session_id = info_.pop("session_id", "")

        # 将抽取结果包装为统一记忆对象，供 text memory 存储层写入。
        return TextualMemoryItem(
            # memory 保存最终可检索、可展示的记忆文本。
            memory=value,
            # metadata 保存与该记忆相关的身份、状态、向量和来源信息。
            metadata=TreeNodeTextualMemoryMetadata(
                user_id=user_id,
                session_id=session_id,
                # 文档 chunk 默认沉淀为长期记忆。
                memory_type="LongTermMemory",
                # 新建节点默认处于 activated 状态，表示可参与检索。
                status="activated",
                tags=tags,
                key=key,
                embedding=embedding,
                # 初始化 usage 为空，后续检索或使用时再追加使用记录。
                usage=[],
                # 保留文档来源信息，让记忆能够追溯到原始文件或文本。
                sources=source_info,
                # 文档 helper 当前不额外写背景摘要，因此置为空字符串。
                background="",
                # 给抽取结果设置默认高置信度，表示当前流程认为它基本可信。
                confidence=0.99,
                # 将节点类型标记为事实类记忆。
                type="fact",
                # 把除 user_id/session_id 之外的附加信息继续保留在 metadata.info 中。
                info=info_,
            ),
        )
    # 捕获 LLM 调用异常，避免文档并发处理时单个任务拖垮整体。
    except Exception as e:
        # 记录构造 TextualMemoryItem 阶段的异常，例如 embedding 或字段校验失败。
        logger.error(f"[BuildNode] Error building node: {e}")
        # 返回 None 表示当前分支没有可用结果或选择安全降级。
        return None


# SimpleStructMemReader 是核心 reader，实现从对话或文档中抽取简单结构化文本记忆。
class SimpleStructMemReader(BaseMemReader, ABC):
    """Naive implementation of MemReader."""

    # 构造函数根据配置集中初始化 LLM、通用 LLM、向量模型、chunker 和运行期依赖占位。
    def __init__(self, config: SimpleStructMemReaderConfig):
        """
        Initialize the NaiveMemReader with configuration.

        Args:
            config: Configuration object for the reader
        """
        # 保存配置对象，后续所有模型、chunker 和开关都从这里读取。
        self.config = config
        # Main LLM for chat/doc memory extraction (fine-tuned model)
        # 根据配置创建主 LLM 实例。
        self.llm = LLMFactory.from_config(config.llm)
        # General LLM for non-chat/doc tasks (hallucination filter, rewrite, merge, etc.)
        # Falls back to main llm if not configured
        # 初始化通用 LLM，并在缺失配置时采用 fallback 策略。
        self.general_llm = (
            # 有单独配置时按该配置创建通用 LLM。
            LLMFactory.from_config(config.general_llm)
            # 只有明确配置 general_llm 时才创建独立实例。
            if config.general_llm is not None
            # 没有通用 LLM 配置时复用主 LLM。
            else self.llm
        )
        # 尝试读取偏好抽取专用 LLM 配置；旧配置不存在时返回 None。
        preference_extractor_llm_config = getattr(config, "preference_extractor_llm", None)
        # 初始化偏好抽取 LLM，优先使用专用模型，否则复用通用模型。
        self.preference_extractor_llm = (
            # 存在专用偏好抽取配置时按该配置创建 LLM。
            LLMFactory.from_config(preference_extractor_llm_config)
            # 只有配置存在时才走专用偏好模型路径。
            if preference_extractor_llm_config is not None
            # 没有专用配置时使用 general_llm，避免功能不可用。
            else self.general_llm
        )
        # 先把 Qwen LLM 置空，表示默认不启用该可选模型。
        self.qwen_llm = None
        # 读取可选的 Qwen 模型配置。
        qwen_llm_config = getattr(config, "qwen_llm", None)
        # 只有配置存在时才尝试初始化 Qwen。
        if qwen_llm_config:
            # 进入受保护代码块，保证局部失败不会直接中断整体流程。
            try:
                # 兼容 dict 形式配置，需要先转成正式配置对象。
                if isinstance(qwen_llm_config, dict):
                    # 用配置工厂校验并标准化 Qwen 配置。
                    qwen_llm_config = LLMConfigFactory.model_validate(qwen_llm_config)
                # 按标准化后的配置创建 Qwen LLM 实例。
                self.qwen_llm = LLMFactory.from_config(qwen_llm_config)
            # 捕获异常并走日志或降级路径。
            except Exception as e:
                # Qwen 是可选能力，初始化失败只警告，不影响主 reader 启动。
                logger.warning(f"[LLM] Qwen initialization failed: {e}")
        # 初始化 embedding 模型，用于把记忆文本转成向量。
        self.embedder = EmbedderFactory.from_config(config.embedder)
        # 初始化文档切分器，用于 doc 模式下把长文本拆成 chunk。
        self.chunker = ChunkerFactory.from_config(config.chunker)
        # 读取是否保存原始文件节点的开关，供上游写库流程使用。
        self.save_rawfile = self.chunker.config.save_rawfile
        # 设置单条记忆最大长度的默认上限。
        self.memory_max_length = 8000
        # Use token-based windowing; default to ~5000 tokens if not configured
        # 从配置读取 chat 窗口 token 上限，缺省为 1024。
        self.chat_window_max_tokens = getattr(self.config, "chat_window_max_tokens", 1024)
        # 绑定 token 计数函数，后续窗口切分统一通过它估算长度。
        self._count_tokens = count_tokens_text
        # searcher 延迟注入，因为 reader 初始化时未必已经构建好检索层。
        self.searcher = None
        # Initialize graph_db as None, can be set later via set_graph_db for
        # recall operations
        # 初始化 graph_db 为空，避免未注入前访问未定义属性。
        self.graph_db = None

    # 注入 graph_db，供后续召回、图关系或记忆版本操作使用。
    def set_graph_db(self, graph_db: "BaseGraphDB | None") -> None:
        self.graph_db = graph_db

    # 注入 searcher，供记忆检索、相关记忆查找或后处理逻辑使用。
    def set_searcher(self, searcher: "Searcher | None") -> None:
        self.searcher = searcher

    # 把抽取出的文本 value 和上下文 info 封装成统一的 TextualMemoryItem。
    def _make_memory_item(
        self,
        # value 是最终要沉淀为记忆的文本内容。
        value: str,
        # info 提供 user_id、session_id 以及额外业务元信息。
        info: dict,
        # memory_type 标识记忆类别。
        memory_type: str,
        # tags 是可选标签列表，用于分类或检索。
        tags: list[str] | None = None,
        # key 是可选记忆摘要键。
        key: str | None = None,
        # sources 记录记忆来源。
        sources: list | None = None,
        # background 保存该记忆的背景摘要。
        background: str = "",
        # type_ 表示更细粒度的记忆属性，默认事实。
        type_: str = "fact",
        # confidence 表示该记忆的可信程度。
        confidence: float = 0.99,
        # need_embed 控制是否立即生成 embedding。
        need_embed: bool = True,
        # 允许额外 metadata 字段透传，例如 manager_user_id、project_id。
        **kwargs,
    ) -> TextualMemoryItem:
        """construct memory item"""
        # 复制 info，避免构造 metadata 时破坏外部共享字典。
        info_ = info.copy()
        # 把 user_id 从附加信息中提升到 metadata 顶层字段。
        user_id = info_.pop("user_id", "")
        # 把 session_id 从附加信息中提升到 metadata 顶层字段。
        session_id = info_.pop("session_id", "")
        # 返回统一的文本记忆实体，存储层和检索层都围绕这个结构工作。
        return TextualMemoryItem(
            # 写入实际记忆文本。
            memory=value,
            # 创建树节点文本记忆的 metadata。
            metadata=TreeNodeTextualMemoryMetadata(
                # 记录记忆所属用户。
                user_id=user_id,
                # 记录记忆来源会话。
                session_id=session_id,
                # 保留调用方判断出的记忆类型，例如 UserMemory 或 LongTermMemory。
                memory_type=memory_type,
                # 新建记忆默认激活，可参与检索。
                status="activated",
                # 没有标签时使用空列表，保证字段类型稳定。
                tags=tags or [],
                # 优先使用传入 key；没有 key 时根据文本自动派生一个。
                key=key if key is not None else derive_key(value),
                # 按需生成 embedding；某些中间转换可以选择暂不向量化。
                embedding=self.embedder.embed([value])[0] if need_embed else None,
                # 初始化使用记录为空。
                usage=[],
                # 记录该记忆来自哪些消息或文档片段。
                sources=sources or [],
                # 保存 LLM 给出的背景摘要或上下文说明。
                background=background,
                # 保存置信度，后续可用于排序或过滤。
                confidence=confidence,
                # 保存事实/偏好等更细粒度类型。
                type=type_,
                # 保留除了 user_id/session_id 外的业务附加字段。
                info=info_,
                # 允许额外 metadata 字段透传，例如 manager_user_id、project_id。
                **kwargs,
            ),
        )

    # 对 LLM 调用做安全包装，避免单次生成失败直接打断整条记忆读取流程。
    def _safe_generate(self, messages: list[dict]) -> str | None:
        # LLM 调用可能因为服务、网络或模型异常失败，因此必须保护。
        try:
            # 正常情况下直接返回主 LLM 的原始生成文本。
            return self.llm.generate(messages)
        # 捕获所有生成异常，保证上层可以继续走降级路径。
        except Exception:
            # 使用 exception 记录堆栈，方便定位 LLM 调用失败原因。
            logger.exception("[LLM] Generation failed")
            # 返回 None 表示当前分支没有可用结果或选择安全降级。
            return None

    # 对 LLM 返回文本做安全 JSON 解析，失败时返回 None 交给上层降级。
    def _safe_parse(self, text: str | None) -> dict | None:
        # 没有文本就没有可解析内容，直接返回 None。
        if not text:
            # 返回 None 表示当前分支没有可用结果或选择安全降级。
            return None
        # 解析模型输出存在格式不稳定风险，因此单独保护。
        try:
            # 尝试把模型输出解析成 dict。
            return parse_json_result(text)
        # 解析异常不向外抛出，交给调用方使用 fallback。
        except Exception:
            # 记录解析失败，提示模型输出可能不满足 JSON 约束。
            logger.warning("[LLM] JSON parse failed")
            # 返回 None 表示当前分支没有可用结果或选择安全降级。
            return None

    # 根据输入语言和自定义标签拼装 prompt，并统一获得可解析的 LLM 结构化响应。
    def _get_llm_response(self, mem_str: str, custom_tags: list[str] | None) -> dict:
        # 先判断输入语言，用于选择中文或英文 prompt。
        lang = detect_lang(mem_str)
        # 根据语言取 chat 抽取模板。
        template = PROMPT_DICT["chat"][lang]
        # 取同语言示例，后续可能按配置移除。
        examples = PROMPT_DICT["chat"][f"{lang}_example"]
        # 把真实对话文本填入 prompt。
        prompt = template.replace("${conversation}", mem_str)

        # 如果用户指定自定义标签，就构造标签约束提示。
        custom_tags_prompt = (
            # 把标签列表写入对应语言的 custom_tags 指令中。
            PROMPT_DICT["custom_tags"][lang].replace("{custom_tags}", str(custom_tags))
            # 只有存在自定义标签时才添加该段指令。
            if custom_tags
            # 没有标签时使用空字符串，避免 prompt 中残留占位内容。
            else ""
        )
        # 把标签约束插入主 prompt。
        prompt = prompt.replace("${custom_tags_prompt}", custom_tags_prompt)

        # 如果配置要求更短 prompt，就移除示例部分。
        if self.config.remove_prompt_example:
            # 从 prompt 中删除示例，降低 token 成本。
            prompt = prompt.replace(examples, "")
        # 封装为 chat completion 风格消息，交给 LLM 生成。
        messages = [{"role": "user", "content": prompt}]

        # 安全调用 LLM，避免异常直接中断。
        response_text = self._safe_generate(messages)
        # 安全解析 LLM 结果。
        response_json = self._safe_parse(response_text)

        # 如果生成或解析失败，就构造一个保底记忆，避免 add 接口看似成功但无写入。
        if not response_json:
            # NOTE: the key MUST be ``"memory list"`` (with a space) — the
            # downstream consumers in ``_process_chat_data`` /
            # ``_process_transfer_chat_data`` read via
            # ``resp.get("memory list", [])``. A typo here drops the
            # salvaged item silently and causes ``/product/add`` to return
            # 200 with zero memories written to Neo4j (bug #1355).
            # 返回符合下游预期结构的 fallback dict。
            return {
                # 下游固定读取 memory list，因此 fallback 必须保持这个键名。
                "memory list": [
                    {
                        # 用输入前十个字符作为简易 key。
                        "key": mem_str[:10],
                        # fallback 默认作为用户记忆保存，避免丢失用户原始表达。
                        "memory_type": "UserMemory",
                        # 把原始输入作为记忆内容保留下来。
                        "value": mem_str,
                        # fallback 不生成标签。
                        "tags": [],
                    }
                ],
                # summary 也使用原始输入，保证字段完整。
                "summary": mem_str,
            }

        # 解析成功时直接返回 LLM 的结构化结果。
        return response_json

    # 按 token 数把较长对话切成滑动窗口，同时保留每行消息来源，方便后续追溯。
    def _iter_chat_windows(self, scene_data_info, max_tokens=None, overlap=200):
        """
        use token counter to get a slide window generator
        """
        # 使用传入上限；没有传入时使用配置里的 chat 窗口上限。
        max_tokens = max_tokens or self.chat_window_max_tokens
        # buf 存窗口文本，sources 存每行来源，start_idx 记录当前窗口起点。
        buf, sources, start_idx = [], [], 0
        # 维护当前窗口拼接后的文本，便于快速估算 token 数。
        cur_text = ""
        # 逐条扫描标准化后的对话消息。
        for idx, item in enumerate(scene_data_info):
            # 读取消息角色，用于在窗口文本中保留说话人。
            role = item.get("role", "")
            # 读取消息正文。
            content = item.get("content", "")
            # 读取可选时间戳，方便后续追溯。
            chat_time = item.get("chat_time", None)
            # 先收集前缀片段，再拼接成一行文本。
            parts = []
            # 有明确角色且不是混合角色时，才把角色写入行前缀。
            if role and str(role).lower() != "mix":
                # 把角色写成类似 user: 的前缀。
                parts.append(f"{role}: ")
            # 如果有消息时间，也加入窗口文本。
            if chat_time:
                # 把时间戳放在角色后面，形成可读上下文。
                parts.append(f"[{chat_time}]: ")
            # 拼出完整行前缀。
            prefix = "".join(parts)
            # 把前缀和正文合成一行，并追加换行。
            line = f"{prefix}{content}\n"

            # 如果加入当前行会超过 token 上限，并且窗口已有内容，就先产出当前窗口。
            if self._count_tokens(cur_text + line) > max_tokens and cur_text:
                # 把窗口缓存合并成完整文本。
                text = "".join(buf)
                # 产出窗口文本和来源快照；copy 防止后续修改影响已产出结果。
                yield {"text": text, "sources": sources.copy(), "start_idx": start_idx}
                # 为了保留上下文重叠，持续丢弃窗口开头，直到剩余内容不超过 overlap。
                while buf and self._count_tokens("".join(buf)) > overlap:
                    # 移除最早的一行文本。
                    buf.pop(0)
                    # 同步移除对应来源，保持文本和来源对齐。
                    sources.pop(0)
                # 把新窗口起点更新为当前消息位置。
                start_idx = idx
                # 每次追加后刷新窗口文本缓存。
                cur_text = "".join(buf)

            # 把当前消息加入窗口缓存。
            buf.append(line)
            # 同时记录当前消息的来源元数据。
            sources.append(
                {
                    # 标记来源类型为 chat。
                    "type": "chat",
                    # 记录原始消息在场景中的下标。
                    "index": idx,
                    # 记录消息角色。
                    "role": role,
                    # 记录消息时间。
                    "chat_time": chat_time,
                    # 记录原始内容，便于追溯和调试。
                    "content": content,
                }
            )
            # 每次追加后刷新窗口文本缓存。
            cur_text = "".join(buf)

        # 循环结束后如果还有未产出的内容，需要作为最后一个窗口返回。
        if buf:
            # 产出最后一个窗口，避免尾部消息丢失。
            yield {"text": "".join(buf), "sources": sources.copy(), "start_idx": start_idx}

    # 对下面的方法统计耗时，便于观察记忆抽取链路中的慢点。
    @timed
    # 处理单个标准化 chat 场景：先切窗口，再按 fast/fine 模式构建记忆节点。
    def _process_chat_data(self, scene_data_info, info, **kwargs):
        # 读取处理模式，默认 fine；fast 更快但不调用 LLM 深度理解。
        mode = kwargs.get("mode", "fine")
        # 先把对话切成 token 受控窗口，后续每个窗口独立抽取记忆。
        windows = list(self._iter_chat_windows(scene_data_info))
        # 从 info 中取出 custom_tags，避免它被写入 metadata.info。
        custom_tags = info.pop(
            # 没有自定义标签时返回 None。
            "custom_tags", None
        # 原注释说明 custom_tags 只服务 prompt，不应作为普通 info 入库。
        )  # must pop here, avoid add to info, only used in sync fine mode

        # 读取调用方传入的用户上下文。
        user_context: UserContext | None = kwargs.get("user_context")
        # 准备额外 metadata 参数，例如 manager_user_id/project_id。
        ctx_kwargs: dict[str, Any] = {}
        # 只有上下文存在时才提取扩展字段。
        if user_context:
            # 如果存在管理者用户 ID，就随记忆一起保存。
            if user_context.manager_user_id:
                # 透传 manager_user_id 到 memory metadata。
                ctx_kwargs["manager_user_id"] = user_context.manager_user_id
            # 如果存在项目 ID，也随记忆保存。
            if user_context.project_id:
                # 透传 project_id 到 memory metadata。
                ctx_kwargs["project_id"] = user_context.project_id

        # fast 模式不调用抽取 LLM，而是把窗口文本直接封装成记忆。
        if mode == "fast":
            # 记录当前使用 fast 流程。
            logger.debug("Using unified Fast Mode")

            # 定义窗口到记忆节点的局部转换函数，便于在线程池中并发执行。
            def _build_fast_node(w):
                text = w["text"]
                roles = {s.get("role", "") for s in w["sources"] if s.get("role")}
                mem_type = "UserMemory" if roles == {"user"} else "LongTermMemory"
                tags = ["mode:fast"]
                # 返回当前函数处理得到的结果。
                return self._make_memory_item(
                    value=text,
                    info=info,
                    memory_type=mem_type,
                    tags=tags,
                    sources=w["sources"],
                    **ctx_kwargs,
                )

            # 进入上下文管理器，确保资源在使用后被正确释放。
            with ContextThreadPoolExecutor(max_workers=8) as ex:
                futures = {ex.submit(_build_fast_node, w): i for i, w in enumerate(windows)}
                results = [None] * len(futures)
                # 遍历当前集合，逐项执行后续处理逻辑。
                for fut in concurrent.futures.as_completed(futures):
                    i = futures[fut]
                    # 进入受保护代码块，保证局部失败不会直接中断整体流程。
                    try:
                        node = fut.result()
                        # 根据当前条件选择是否进入这个处理分支。
                        if node:
                            results[i] = node
                    # 捕获异常并走日志或降级路径。
                    except Exception as e:
                        logger.error(f"[ChatFast] error: {e}")
                chat_nodes = [r for r in results if r]
            # 返回当前函数处理得到的结果。
            return chat_nodes
        # 进入与前面条件相反的处理分支。
        else:
            logger.debug("Using unified Fine Mode")
            chat_read_nodes = []
            # 遍历当前集合，逐项执行后续处理逻辑。
            for w in windows:
                resp = self._get_llm_response(w["text"], custom_tags)
                # 遍历当前集合，逐项执行后续处理逻辑。
                for m in resp.get("memory list", []):
                    # 进入受保护代码块，保证局部失败不会直接中断整体流程。
                    try:
                        memory_type = (
                            m.get("memory_type", "LongTermMemory")
                            .replace("长期记忆", "LongTermMemory")
                            .replace("用户记忆", "UserMemory")
                        )
                        node = self._make_memory_item(
                            value=m.get("value", ""),
                            info=info,
                            memory_type=memory_type,
                            tags=m.get("tags", []),
                            key=m.get("key", ""),
                            sources=w["sources"],
                            background=resp.get("summary", ""),
                            **ctx_kwargs,
                        )
                        chat_read_nodes.append(node)
                    # 捕获异常并走日志或降级路径。
                    except Exception as e:
                        logger.error(f"[ChatFine] parse error: {e}")
            # 返回当前函数处理得到的结果。
            return chat_read_nodes

    # 把已有 Raw/旧记忆重新送入 chat 抽取逻辑，转换成 SimpleStruct 记忆节点。
    def _process_transfer_chat_data(
        # 该方法接收旧文档记忆节点，计划转换为 simple memory。
        self, raw_node: TextualMemoryItem, custom_tags: list[str] | None = None, **kwargs
    ):
        # 取出已有节点的原始记忆文本，准备重新抽取。
        raw_memory = raw_node.memory
        # 对原始记忆再次调用 chat prompt，得到标准结构化结果。
        response_json = self._get_llm_response(raw_memory, custom_tags)

        # 读取上下文，便于迁移时保留管理者或项目字段。
        user_context: UserContext | None = kwargs.get("user_context")
        # 准备额外 metadata 字段。
        ctx_kwargs: dict[str, Any] = {}
        # 根据当前条件选择是否进入这个处理分支。
        if user_context:
            # 根据当前条件选择是否进入这个处理分支。
            if user_context.manager_user_id:
                ctx_kwargs["manager_user_id"] = user_context.manager_user_id
            # 根据当前条件选择是否进入这个处理分支。
            if user_context.project_id:
                ctx_kwargs["project_id"] = user_context.project_id

        # 收集迁移后生成的新记忆节点。
        chat_read_nodes = []
        # 遍历重新抽取出的候选记忆。
        for memory_i_raw in response_json.get("memory list", []):
            # 进入受保护代码块，保证局部失败不会直接中断整体流程。
            try:
                memory_type = (
                    # 读取候选记忆类型，缺省为长期记忆。
                    memory_i_raw.get("memory_type", "LongTermMemory")
                    .replace("长期记忆", "LongTermMemory")
                    .replace("用户记忆", "UserMemory")
                )
                # 防御模型返回未知类型，避免写入不受支持的 memory_type。
                if memory_type not in ["LongTermMemory", "UserMemory"]:
                    # 未知类型统一降级为 LongTermMemory。
                    memory_type = "LongTermMemory"
                # 构造迁移后的新 TextualMemoryItem。
                node_i = self._make_memory_item(
                    value=memory_i_raw.get("value", ""),
                    # 重新组装 info，保留旧节点 info 并补回必要身份字段。
                    info={
                        # 继承旧节点 metadata.info。
                        **(raw_node.metadata.info or {}),
                        # 从旧节点 metadata 中恢复 user_id。
                        "user_id": raw_node.metadata.user_id,
                        # 从旧节点 metadata 中恢复 session_id。
                        "session_id": raw_node.metadata.session_id,
                    },
                    memory_type=memory_type,
                    # 优先使用模型输出的 tags。
                    tags=memory_i_raw.get("tags", [])
                    # 只接受 list 类型 tags。
                    if isinstance(memory_i_raw.get("tags", []), list)
                    # 如果 tags 类型不对，则降级为空列表。
                    else [],
                    key=memory_i_raw.get("key", ""),
                    # 沿用旧节点的来源信息，保持可追溯。
                    sources=raw_node.metadata.sources,
                    background=response_json.get("summary", ""),
                    # 迁移生成的节点默认标为事实记忆。
                    type_="fact",
                    # 设置默认置信度。
                    confidence=0.99,
                    **ctx_kwargs,
                )
                # 保存迁移成功的新节点。
                chat_read_nodes.append(node_i)
            # 捕获异常并走日志或降级路径。
            except Exception as e:
                # 记录某条迁移候选记忆构造失败。
                logger.error(f"[ChatReader] Error parsing memory item: {e}")

        # 返回迁移后的记忆节点列表。
        return chat_read_nodes

    # 外部主入口：校验输入、兼容旧格式、标准化 scene_data，然后分派到内部读取流程。
    def get_memory(
        self,
        # scene_data 是外部传入的 chat/doc 场景数据。
        scene_data: SceneDataInput,
        # type 指明 scene_data 是 chat 还是 doc。
        type: str,
        # info 保存用户和会话身份以及额外字段。
        info: dict[str, Any],
        # mode 控制使用 fast 还是 fine 抽取流程。
        mode: str = "fine",
        # user_name 预留给数据库写入或召回逻辑使用。
        user_name: str | None = None,
        # kwargs 透传上层上下文字段和扩展参数。
        **kwargs,
    ) -> list[list[TextualMemoryItem]]:
        """
        Extract and classify memory content from scene_data.
        For dictionaries: Use LLM to summarize pairs of Q&A
        For file paths: Use chunker to split documents and LLM to summarize each chunk

        Args:
            scene_data: List of dialogue information or document paths
            type: (Deprecated) not supported in the future. Type of scene_data: ['doc', 'chat']
            info: Dictionary containing user_id and session_id.
                Must be in format: {"user_id": "1111", "session_id": "2222"}
                Optional parameters:
                - topic_chunk_size: Size for large topic chunks (default: 1024)
                - topic_chunk_overlap: Overlap for large topic chunks (default: 100)
                - chunk_size: Size for small chunks (default: 256)
                - chunk_overlap: Overlap for small chunks (default: 50)
            mode: mem-reader mode, fast for quick process while fine for
            better understanding via calling llm
            user_name: tha user_name would be inserted later into the
            database, may be used in recall.
        Returns:
            list[list[TextualMemoryItem]] containing memory content with summaries as keys and original text as values
        Raises:
            ValueError: If scene_data is empty or if info dictionary is missing required fields
        """
        # 入口首先拒绝空输入，避免后续流程生成无意义结果。
        if not scene_data:
            # 明确告知调用方 scene_data 不能为空。
            raise ValueError("scene_data is empty")

        # Validate info dictionary format
        # info 必须是字典，因为后面会用 key 访问 user_id/session_id。
        if not isinstance(info, dict):
            # info 类型错误时直接抛出参数异常。
            raise ValueError("info must be a dictionary")

        # 定义最小必需字段集合。
        required_fields = {"user_id", "session_id"}
        # 计算调用方漏传了哪些必需字段。
        missing_fields = required_fields - set(info.keys())
        # 如果有缺失字段，不能继续构建 metadata。
        if missing_fields:
            # 把缺失字段返回给调用方，方便修正请求。
            raise ValueError(f"info dictionary is missing required fields: {missing_fields}")

        # user_id/session_id 必须是字符串，保证 metadata 类型稳定。
        if not all(isinstance(info[field], str) for field in required_fields):
            # 身份字段类型错误时立即抛错。
            raise ValueError("user_id and session_id must be strings")

        # Backward compatibility, after coercing scene_data, we only tackle
        # with standard scene_data type: MessagesType
        # 将旧版 list[str]/list[dict] 等输入统一转换为标准 scene_data。
        standard_scene_data = coerce_scene_data(scene_data, type)
        # 把标准化后的数据交给内部读取流程。
        return self._read_memory(
            standard_scene_data, type, info, mode, user_name=user_name, **kwargs
        )

    # 使用通用 LLM 判断并改写记忆文本，减少表述不完整或不够贴近上下文的问题。
    def rewrite_memories(
        self, messages: list[dict], memory_list: list[TextualMemoryItem], user_only: bool = True
    ) -> list[TextualMemoryItem]:
        # Build input objects with memory text and metadata (timestamps, sources, etc.)
        # user_only 模式只参考非 assistant 消息，避免助手回复影响用户事实记忆。
        if user_only:
            # 选择只基于用户消息的改写 prompt。
            template = PROMPT_MAPPING["rewrite_user_only"]
            # 过滤掉 assistant 消息，只保留用户/系统等上下文。
            filtered_messages = [m for m in messages if m.get("role") != "assistant"]
            # 如果过滤后没有上下文，就没有依据改写。
            if len(filtered_messages) < 1:
                # 兜底返回原列表，避免改写模块影响主链路。
                return memory_list
        # 不需要改写时保留原记忆。
        else:
            # 选择通用改写 prompt。
            template = PROMPT_MAPPING["rewrite"]
            # 保留全部消息作为改写依据。
            filtered_messages = messages
            # 完整改写至少需要较完整上下文，否则不执行。
            if len(filtered_messages) < 2:
                # 兜底返回原列表，避免改写模块影响主链路。
                return memory_list

        # 准备填充 prompt 的两个核心变量：消息上下文和候选记忆。
        prompt_args = {
            # 把消息按行展开，便于 LLM 阅读对话上下文。
            "messages_inline": "\n".join(
                # 每条消息以 role+content 的形式写入 prompt。
                [f"- [{message['role']}]: {message['content']}" for message in filtered_messages]
            ),
            # 把候选记忆序列化成 JSON，便于 LLM 按索引返回判断。
            "memories_inline": json.dumps(
                # 用索引映射到记忆文本，方便后续根据索引替换。
                {idx: mem.memory for idx, mem in enumerate(memory_list)},
                # 保留中文原文，不转义成 Unicode。
                ensure_ascii=False,
                # 格式化 JSON，提高 prompt 可读性。
                indent=2,
            ),
        }
        # 将上下文和记忆列表填入改写模板。
        prompt = template.format(**prompt_args)

        # Optionally run filter and parse the output
        # Use general_llm for rewrite (not fine-tuned for this task)
        # 进入受保护代码块，保证局部失败不会直接中断整体流程。
        try:
            # 把构造好的 prompt 交给通用 LLM。
            raw = self.general_llm.generate([{"role": "user", "content": prompt}])
            # 解析 LLM 的改写结果，期望得到按记忆索引组织的 dict。
            success, parsed = parse_rewritten_response(raw)
            logger.info(
                f"[rewrite_memories] Hallucination filter parsed successfully: {success}；prompt: {prompt}"
            )
            # 只有解析成功才按 LLM 决策修改记忆。
            if success:
                logger.info(f"Rewrite filter result: {parsed}")

                # 准备收集改写后的记忆列表。
                new_memory_list = []
                # 遍历每条 LLM 返回的改写决策。
                for mem_idx, content in parsed.items():
                    # 防御 LLM 返回越界索引。
                    if mem_idx < 0 or mem_idx >= len(memory_list):
                        logger.warning(
                            f"[rewrite_memories] Invalid memory index {mem_idx} for memory_list {len(memory_list)}, skipping."
                        )
                        # 跳过当前无效或无法处理的条目。
                        continue

                    # 读取是否需要改写的判断。
                    need_rewrite = content.get("need_rewrite", False)
                    # 读取改写后的文本。
                    rewritten_text = content.get("rewritten", "")
                    # 读取改写原因，用于日志审计。
                    reason = content.get("reason", "")
                    # 保留原文本，便于日志对比。
                    original_text = memory_list[mem_idx].memory

                    # Replace memory text with rewritten content when rewrite is needed
                    # 确认需要改写，并且改写内容是字符串。
                    if need_rewrite and isinstance(rewritten_text, str):
                        logger.info(
                            f"[rewrite_memories] index={mem_idx}, need_rewrite={need_rewrite}, rewritten='{rewritten_text}', reason='{reason}', original memory='{original_text}', action='replace_text'"
                        )
                        # 避免用空字符串覆盖原记忆。
                        if len(rewritten_text.strip()) != 0:
                            # 原地更新该条记忆文本。
                            memory_list[mem_idx].memory = rewritten_text
                            # 把改写后的记忆加入新列表。
                            new_memory_list.append(memory_list[mem_idx])
                    # 不需要改写时保留原记忆。
                    else:
                        # 把改写后的记忆加入新列表。
                        new_memory_list.append(memory_list[mem_idx])
                # 返回经过改写决策处理后的列表。
                return new_memory_list
            # 不需要改写时保留原记忆。
            else:
                # 解析失败时记录 warning，并回退到原列表。
                logger.warning("Rewrite filter parsing failed or returned empty result.")
        # 捕获异常并走日志或降级路径。
        except Exception as e:
            # LLM 或解析流程异常时记录错误堆栈。
            logger.error(f"Rewrite filter execution error: {e}", stack_info=True)

        # 兜底返回原列表，避免改写模块影响主链路。
        return memory_list

    # 使用通用 LLM 对候选记忆做幻觉过滤，只保留能被原始消息支撑的内容。
    def filter_hallucination_in_memories(
        self, messages: list[dict], memory_list: list[TextualMemoryItem]
    ) -> list[TextualMemoryItem]:
        # Build input objects with memory text and metadata (timestamps, sources, etc.)
        # 选择幻觉过滤 prompt，让 LLM 判断记忆是否被原始消息支持。
        template = PROMPT_MAPPING["hallucination_filter"]
        # 上下文过短时过滤依据不足。
        if len(messages) < 2:
            # 依据不足时原样保留所有记忆。
            return memory_list
        # 准备 prompt 参数：原始消息和候选记忆。
        prompt_args = {
            "messages_inline": "\n".join(
                [f"- [{message['role']}]: {message['content']}" for message in messages]
            ),
            "memories_inline": json.dumps(
                {idx: mem.memory for idx, mem in enumerate(memory_list)},
                ensure_ascii=False,
                indent=2,
            ),
        }
        # 生成最终过滤 prompt。
        prompt = template.format(**prompt_args)

        # Optionally run filter and parse the output
        # Use general_llm for hallucination filter (not fine-tuned for this task)
        # 进入受保护代码块，保证局部失败不会直接中断整体流程。
        try:
            # 调用通用 LLM 进行保留/删除判断。
            raw = self.general_llm.generate([{"role": "user", "content": prompt}])
            # 解析 LLM 返回的 keep/reason 结构。
            success, parsed = parse_keep_filter_response(raw)
            logger.info(
                f"[filter_hallucination_in_memories] Hallucination filter parsed successfully: {success}；prompt: {prompt}"
            )
            # 解析成功时才执行过滤。
            if success:
                logger.info(f"Hallucination filter result: {parsed}")

                # 准备收集保留下来的记忆。
                filtered_list = []
                # 逐条检查候选记忆。
                for mem_idx, mem in enumerate(memory_list):
                    # 根据索引读取 LLM 对该条记忆的判断。
                    content = parsed.get(mem_idx)
                    # 如果 LLM 漏判该条记忆，为安全起见选择保留。
                    if not content:
                        # 记录漏判情况，方便后续评估 prompt 稳定性。
                        logger.warning(f"No verdict for memory {mem_idx}, keeping it.")
                        # 把该记忆保留下来。
                        filtered_list.append(mem)
                        # 跳过当前项，继续处理后续输入。
                        continue

                    # 默认 keep=True，避免解析字段缺失时误删记忆。
                    keep = content.get("keep", True)
                    # 读取删除或保留理由。
                    reason = content.get("reason", "")

                    # LLM 判断可保留时加入结果。
                    if keep:
                        # 把该记忆保留下来。
                        filtered_list.append(mem)
                    # LLM 判断不应保留时只记录日志，不加入结果。
                    else:
                        logger.info(
                            f"[filter_hallucination_in_memories] Dropping memory index={mem_idx}, reason='{reason}', memory='{mem.memory}'"
                        )

                # 返回过滤后的记忆列表。
                return filtered_list
            # LLM 判断不应保留时只记录日志，不加入结果。
            else:
                # 过滤结果解析失败时不做删除。
                logger.warning("Hallucination filter parsing failed or returned empty result.")
        # 捕获异常并走日志或降级路径。
        except Exception as e:
            # 过滤流程异常时记录错误。
            logger.error(f"Hallucination filter execution error: {e}", stack_info=True)

        # 依据不足时原样保留所有记忆。
        return memory_list

    # 内部读取主流程：根据 chat/doc 类型选择处理函数，并用线程池并发处理多个场景。
    def _read_memory(
        self,
        # messages 是已经标准化后的多个 scene。
        messages: list[MessagesType],
        # type 指明 scene_data 是 chat 还是 doc。
        type: str,
        # info 保存用户和会话身份以及额外字段。
        info: dict[str, Any],
        # mode 控制使用 fast 还是 fine 抽取流程。
        mode: str = "fine",
        # kwargs 透传上层上下文字段和扩展参数。
        **kwargs,
    ) -> list[list[TextualMemoryItem]]:
        """
        1. raw file:
        [
            [
                {"type": "file", "file": "str"}
            ],
            [
                {"type": "file", "file": "str"}
            ],...
        ]
        2. text chat:
        scene_data = [
            [ {role: user, ...}, {role: assistant, ...}, ... ],
            [ {role: user, ...}, {role: assistant, ...}, ... ],
            [ ... ]
        ]
        """
        # 先把标准化场景再转换成当前 reader 可处理的分组。
        list_scene_data_info = self.get_scene_data_info(messages, type)

        # 准备收集每个场景处理后的记忆列表。
        memory_list = []
        # chat 类型使用对话处理函数。
        if type == "chat":
            # 绑定 chat 场景处理函数。
            processing_func = self._process_chat_data
        # doc 类型使用文档处理函数。
        elif type == "doc":
            # 绑定 doc 场景处理函数。
            processing_func = self._process_doc_data
        # 未知类型默认按 doc 处理，兼容历史路径。
        else:
            # 绑定 doc 场景处理函数。
            processing_func = self._process_doc_data

        # Process Q&A pairs concurrently with context propagation
        # 创建线程池，让多个 scene 并行抽取。
        with ContextThreadPoolExecutor() as executor:
            # 收集所有并发任务 future。
            futures = [
                # 把每个 scene 提交给对应处理函数。
                executor.submit(processing_func, scene_data_info, info, mode=mode)
                # 遍历所有整理后的 scene 分组。
                for scene_data_info in list_scene_data_info
            ]
            # 按完成顺序收集处理结果。
            for future in concurrent.futures.as_completed(futures):
                # 进入受保护代码块，保证局部失败不会直接中断整体流程。
                try:
                    # 读取单个 scene 的记忆抽取结果。
                    res_memory = future.result()
                    # 只要不是 None，就认为该 scene 有可合并结果。
                    if res_memory is not None:
                        # 把该 scene 的记忆列表加入总列表。
                        memory_list.append(res_memory)
                # 捕获异常并走日志或降级路径。
                except Exception as e:
                    # 记录单个 scene 任务失败。
                    logger.error(f"Task failed with exception: {e}")
                    # 额外输出完整堆栈，方便排查并发任务内部错误。
                    logger.error(traceback.format_exc())

        # 读取环境变量开关，决定是否对新增记忆启用幻觉过滤。
        if os.getenv("SIMPLE_STRUCT_ADD_FILTER", "false") == "true":
            # Build inputs
            # 准备把多个 scene 的消息合并成一个上下文列表。
            combined_messages = []
            # 遍历原始标准化消息分组。
            for group_messages in messages:
                # 把每组消息追加到总上下文。
                combined_messages.extend(group_messages)

            # 逐组处理抽取出的记忆列表。
            for group_id in range(len(memory_list)):
                # 进入受保护代码块，保证局部失败不会直接中断整体流程。
                try:
                    # 深拷贝当前组，确保过滤/改写尝试不会先污染原结果。
                    original_memory_group = copy.deepcopy(memory_list[group_id])
                    # 序列化过滤前文本，用于后面判断是否发生变化。
                    serialized_origin_memories = json.dumps(
                        # 只比较 memory 文本，忽略 metadata 差异。
                        [one.memory for one in original_memory_group], indent=2
                    )
                    # 调用幻觉过滤器得到修订后的记忆组。
                    revised_memory_list = self.filter_hallucination_in_memories(
                        messages=combined_messages,
                        memory_list=original_memory_group,
                    )
                    # 序列化过滤后的文本，用于和原结果对比。
                    serialized_revised_memories = json.dumps(
                        [one.memory for one in revised_memory_list], indent=2
                    )
                    # 只有过滤结果有变化时才替换原 memory_list。
                    if serialized_origin_memories != serialized_revised_memories:
                        # 将过滤后的结果写回对应分组。
                        memory_list[group_id] = revised_memory_list
                        logger.info(
                            f"[SIMPLE_STRUCT_ADD_FILTER] Modified the list for group_id={group_id}: "
                            f"\noriginal={serialized_origin_memories},"
                            f"\nrevised={serialized_revised_memories}"
                        )

                # 捕获异常并走日志或降级路径。
                except Exception as e:
                    # 异常时仍尝试把当前组记忆转成可日志化文本。
                    group_serialized = [
                        # 兼容异常数据结构：有 memory 字段就取 memory，否则转字符串。
                        one.memory if hasattr(one, "memory") else str(one)
                        # 遍历当前出错分组。
                        for one in memory_list[group_id]
                    ]
                    logger.error(
                        f"There is an exception while filtering group_id={group_id}: {e}\n"
                        f"messages: {combined_messages}\n"
                        f"memory_list(serialized): {group_serialized}",
                        exc_info=True,
                    )
        # 返回按 scene 分组的记忆结果。
        return memory_list

    # 把已有 TextualMemoryItem 按 fine 模式重新转换成更细粒度的 simple memory。
    def fine_transfer_simple_mem(
        self,
        # input_memories 是待迁移转换的旧记忆节点。
        input_memories: list[TextualMemoryItem],
        # type 指明 scene_data 是 chat 还是 doc。
        type: str,
        # custom_tags 是可选标签约束，用于影响 LLM 抽取结果。
        custom_tags: list[str] | None = None,
        # kwargs 透传上层上下文字段和扩展参数。
        **kwargs,
    ) -> list[list[TextualMemoryItem]]:
        # 没有输入记忆时无需转换。
        if not input_memories:
            # 直接返回空结果。
            return []

        # 准备收集迁移转换后的记忆组。
        memory_list = []

        # chat 类型走 chat 转换处理。
        if type == "chat":
            # 绑定 chat 迁移转换函数。
            processing_func = self._process_transfer_chat_data
        # doc 类型走 doc 转换处理。
        elif type == "doc":
            # 绑定 doc 迁移转换函数。
            processing_func = self._process_transfer_doc_data
        # 未知类型默认按 doc 转换。
        else:
            # 绑定 doc 迁移转换函数。
            processing_func = self._process_transfer_doc_data

        # Process Q&A pairs concurrently with context propagation
        # 使用线程池并行转换多条输入记忆。
        with ContextThreadPoolExecutor() as executor:
            futures = [
                # 把每条原始记忆提交给对应迁移函数。
                executor.submit(processing_func, scene_data_info, custom_tags, **kwargs)
                # 遍历输入记忆列表。
                for scene_data_info in input_memories
            ]
            # 遍历当前集合，逐项执行后续处理逻辑。
            for future in concurrent.futures.as_completed(futures):
                # 进入受保护代码块，保证局部失败不会直接中断整体流程。
                try:
                    # 读取单条迁移任务的返回值。
                    res_memory = future.result()
                    # 非 None 结果才加入最终列表。
                    if res_memory is not None:
                        # 保存该条输入记忆迁移出的新节点列表。
                        memory_list.append(res_memory)
                # 捕获异常并走日志或降级路径。
                except Exception as e:
                    logger.error(f"Task failed with exception: {e}")
                    logger.error(traceback.format_exc())
        # 返回所有迁移结果。
        return memory_list

    # 把标准化后的 scene_data 再整理成当前 reader 真正能处理的 chat/doc 形态。
    def get_scene_data_info(self, scene_data: list, type: str) -> list[list[Any]]:
        """
        Convert normalized MessagesType scenes into typical MessagesType this reader can
        handle.
        SimpleStructMemReader only supports text-only chat messages with roles.
        For chat scenes we:
          - skip unsupported scene types (e.g. `str` scenes)
          - drop non-dict messages
          - keep only roles in {user, assistant, system}
          - coerce OpenAI multimodal `content` (list[parts]) into a single plain-text string
          - then apply the existing windowing logic (<=10 messages with 2-message overlap)
        For doc scenes we pass through; doc handling is done in `_process_doc_data`.
        """
        # 初始化结果列表，每个元素代表一个可独立处理的 scene/window。
        results: list[list[Any]] = []

        # chat 类型需要做角色、内容和窗口校验。
        if type == "chat":
            # 只允许常规 chat 角色，过滤 tool 或未知角色。
            allowed_roles = {"user", "assistant", "system"}
            # 遍历每个标准化后的 chat scene。
            for items in scene_data:
                # 字符串 scene 当前不支持，说明输入仍是旧式或异常格式。
                if isinstance(items, str):
                    logger.warning(
                        "SimpleStruct MemReader does not support "
                        "str message data now, your messages "
                        f"contains {items}, skipping"
                    )
                    # 跳过不符合要求的 scene 或消息。
                    continue
                # 每个 chat scene 应该是消息 dict 列表。
                if not isinstance(items, list):
                    logger.warning(
                        "SimpleStruct MemReader expects message as "
                        f"list[dict], your messages contains"
                        f"{items}, skipping"
                    )
                    # 跳过不符合要求的 scene 或消息。
                    continue
                # Filter messages within this message
                # 收集当前 scene 中有效的标准消息。
                result = []
                # 逐条检查当前 scene 的消息。
                for _i, item in enumerate(items):
                    # 每条消息必须是 dict，才能读取 role/content。
                    if not isinstance(item, dict):
                        logger.warning(
                            "SimpleStruct MemReader expects message as "
                            f"list[dict], your messages contains"
                            f"{item}, skipping"
                        )
                        # 跳过不符合要求的 scene 或消息。
                        continue
                    # 读取 role，缺失时先用空字符串。
                    role = item.get("role") or ""
                    # 把非字符串 role 转成字符串，增强容错性。
                    role = role if isinstance(role, str) else str(role)
                    # 去除空白并转小写，统一角色格式。
                    role = role.strip().lower()
                    # 过滤不在允许集合内的角色。
                    if role not in allowed_roles:
                        logger.warning(
                            f"SimpleStruct MemReader expects message with "
                            f"role in {allowed_roles}, your messages contains"
                            f"role {role}, skipping"
                        )
                        # 跳过不符合要求的 scene 或消息。
                        continue

                    # 读取消息正文。
                    content = item.get("content", "")
                    # 当前 SimpleStructMemReader 只支持纯文本 content。
                    if not isinstance(content, str):
                        logger.warning(
                            f"SimpleStruct MemReader expects message content "
                            f"with str, your messages content"
                            f"is {content!s}, skipping"
                        )
                        # 跳过不符合要求的 scene 或消息。
                        continue
                    # 空文本无法形成有效上下文。
                    if not content:
                        # 跳过不符合要求的 scene 或消息。
                        continue

                    # 把通过校验的消息按统一字段结构加入当前 scene。
                    result.append(
                        {
                            # 保存规范化后的角色。
                            "role": role,
                            # 保存原始文本内容。
                            "content": content,
                            # 保留可选聊天时间，缺失时为空。
                            "chat_time": item.get("chat_time", ""),
                        }
                    )
                # 如果当前 scene 没有任何有效消息，就跳过。
                if not result:
                    # 跳过不符合要求的 scene 或消息。
                    continue
                # 初始化最多 10 条消息的滑动窗口。
                window = []
                # 按顺序把有效消息放入窗口。
                for i, item in enumerate(result):
                    # 当前消息进入窗口。
                    window.append(item)
                    # 窗口达到 10 条后切出一个处理单元。
                    if len(window) >= 10:
                        # 追加尾部窗口。
                        results.append(window)
                        # 如果后面还有消息，就保留最后 2 条作为下一个窗口的上下文重叠。
                        context = copy.deepcopy(window[-2:]) if i + 1 < len(result) else []
                        # 用重叠上下文作为新窗口起点。
                        window = context

                # 循环结束后如果还有未满 10 条的窗口，也需要保留。
                if window:
                    # 追加尾部窗口。
                    results.append(window)
        # doc 类型不做 chat 过滤，直接交给文档处理函数。
        elif type == "doc":
            # 文档场景直接透传。
            results = scene_data
        # 返回整理后的 scene/window 列表。
        return results

    # 处理单个文档场景：抽出文本、切块、为每个块构造 prompt，并并发生成记忆节点。
    def _process_doc_data(self, scene_data_info, info, **kwargs):
        """
        Process doc data after being normalized to new RawMessageList format.

        scene_data_info format (length always == 1):
        [
            {"type": "file", "file": {"filename": "...", "file_data": "..."}}
        ]
        OR
        [
            {"type": "text", "text": "..."}
        ]

        Behavior:
        - Merge all text/file_data into a single "full text"
        - Chunk the text
        - Build prompts
        - Send to LLM
        - Parse results and build memory nodes
        """
        # 读取文档处理模式；当前只支持 fine。
        mode = kwargs.get("mode", "fine")
        # 文档 fast 模式尚未实现。
        if mode == "fast":
            # 显式抛出未实现异常，避免调用方误以为该路径已经可用。
            raise NotImplementedError

        # 取出 custom_tags 仅用于 prompt，不写入普通 metadata.info。
        custom_tags = info.pop("custom_tags", None)

        # 文档标准化后应只有一个 file/text item；否则无法确定文档来源。
        if not scene_data_info or len(scene_data_info) != 1:
            # 记录结构错误，提示上游标准化流程可能不符合预期。
            logger.error(
                "[DocReader] scene_data_info must contain exactly 1 item after normalization"
            )
            # 结构非法或内容为空时返回空列表，避免继续处理。
            return []

        # 取出唯一的文档输入项。
        item = scene_data_info[0]
        # 初始化待切分的完整文本。
        text_content = ""
        # 初始化来源信息列表，后续写入每个记忆节点。
        source_info_list = []

        # Determine content and source metadata
        # 文件输入从 file 字段中取 filename 和 file_data。
        if item.get("type") == "file":
            # 读取文件对象。
            f = item["file"]
            # 读取文件名；缺失时使用 document 作为默认来源名。
            filename = f.get("filename") or "document"
            # 读取文件文本内容；缺失时为空。
            file_data = f.get("file_data") or ""

            # 文件内容就是后续要 chunk 的全文。
            text_content = file_data
            # 构造文档来源元数据。
            source_dict = {
                # 来源类型标记为 doc。
                "type": "doc",
                # 用文件名作为 doc_path，便于结果追溯。
                "doc_path": filename,
            }
            # 把来源 dict 转成 SourceMessage 对象列表。
            source_info_list = [SourceMessage(**source_dict)]

        # 内联文本输入没有文件名，走 text 分支。
        elif item.get("type") == "text":
            # 读取内联文本。
            text_content = item.get("text", "")
            # 给内联文本设置固定来源标识。
            source_info_list = [SourceMessage(type="doc", doc_path="inline-text")]

        # 去除全文首尾空白，并兼容 None。
        text_content = (text_content or "").strip()
        # 空文档没有可抽取内容。
        if not text_content:
            # 记录文档为空，方便排查输入。
            logger.warning("[DocReader] Empty document text after normalization.")
            # 结构非法或内容为空时返回空列表，避免继续处理。
            return []

        # 使用 chunker 把长文档切成适合 LLM 处理的小块。
        chunks = self.chunker.chunk(text_content)
        # 准备保存每个 chunk 对应的 LLM prompt message。
        messages = []
        # 逐个文档 chunk 构造 prompt。
        for chunk in chunks:
            # 根据 chunk 内容选择中英文文档 prompt。
            lang = detect_lang(chunk.text)
            # 获取对应语言的文档抽取模板。
            template = PROMPT_DICT["doc"][lang]
            # 把当前 chunk 文本填入模板。
            prompt = template.replace("{chunk_text}", chunk.text)
            # 按需构造自定义标签约束。
            custom_tags_prompt = (
                PROMPT_DICT["custom_tags"][lang].replace("{custom_tags}", str(custom_tags))
                if custom_tags
                else ""
            )
            # 把自定义标签提示填入文档 prompt。
            prompt = prompt.replace("{custom_tags_prompt}", custom_tags_prompt)
            # 封装为 LLM 消息格式。
            message = [{"role": "user", "content": prompt}]
            # 保存当前 chunk 的 prompt message。
            messages.append(message)

        # 准备收集文档 chunk 生成的记忆节点。
        doc_nodes = []

        # 使用较大的线程池并发处理多个文档 chunk。
        with ContextThreadPoolExecutor(max_workers=50) as executor:
            # 建立 future 到 chunk 索引的映射。
            futures = {
                # 提交单个 chunk 的 _build_node 任务。
                executor.submit(
                    # 每个任务都调用顶层 helper 完成生成、解析和节点构建。
                    _build_node,
                    # 传入 chunk 索引。
                    idx,
                    # 传入当前 chunk 的 LLM message。
                    msg,
                    # 传入用户/session 等上下文信息。
                    info,
                    # 传入文档来源信息。
                    source_info_list,
                    # 传入主 LLM。
                    self.llm,
                    # 传入 JSON 解析函数。
                    parse_json_result,
                    # 传入 embedding 模型。
                    self.embedder,
                ): idx
                # 为每个 chunk message 创建一个并发任务。
                for idx, msg in enumerate(messages)
            }
            # 记录任务总数，用于 tqdm 进度条。
            total = len(futures)

            # 用 tqdm 包装 as_completed，展示 chunk 处理进度。
            for future in tqdm(
                # 按完成顺序消费 future，同时告诉进度条总任务数。
                concurrent.futures.as_completed(futures), total=total, desc="Processing"
            ):
                # 进入受保护代码块，保证局部失败不会直接中断整体流程。
                try:
                    # 读取单个 chunk 生成的节点。
                    node = future.result()
                    # 只保留成功构建的节点。
                    if node:
                        # 把文档记忆节点加入结果。
                        doc_nodes.append(node)
                # 捕获异常并走日志或降级路径。
                except Exception as e:
                    # 通过 tqdm.write 输出错误，避免破坏进度条显示。
                    tqdm.write(f"[ERROR] {e}")
                    # 记录单个文档 chunk 任务失败。
                    logger.error(f"[DocReader] Future task failed: {e}")
        # 返回当前文档抽取出的所有记忆节点。
        return doc_nodes

    # 预留文档迁移转换入口；当前尚未实现。
    def _process_transfer_doc_data(
        # 该方法接收旧文档记忆节点，计划转换为 simple memory。
        self, raw_node: TextualMemoryItem, custom_tags: list[str] | None = None, **kwargs
    ):
        # 显式抛出未实现异常，避免调用方误以为该路径已经可用。
        raise NotImplementedError
