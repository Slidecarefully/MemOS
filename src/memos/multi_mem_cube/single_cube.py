# 延迟解析类型注解，避免运行期因前向引用或 TYPE_CHECKING 导入产生额外依赖。
from __future__ import annotations

# json 用于把请求、任务内容和候选记忆序列化成字符串，便于调度器或 LLM Prompt 使用。
import json
# time 用于记录流程耗时，后面会用 perf_counter 统计各阶段性能。
import time
# traceback 用于在捕获异常时输出完整堆栈，帮助定位搜索流程中的失败原因。
import traceback

# dataclass 自动生成初始化等样板代码，让 SingleCubeView 主要表达依赖和业务方法。
from dataclasses import dataclass
# datetime 用于给调度任务打 UTC 时间戳。
from datetime import datetime
# TYPE_CHECKING 让类型检查所需依赖只在静态分析时导入；Any 用于标注外部组件或插件对象。
from typing import TYPE_CHECKING, Any

# 引入记忆结果格式化相关工具，搜索结果最终会通过这些函数整理成统一响应。
from memos.api.handlers.formatters_handler import (
    # format_memory_item 把内部 TextualMemoryItem 或搜索结果转换成 API 可返回的字典结构。
    format_memory_item,
    # post_process_textual_mem 负责把文本记忆结果归并到统一的 MOSSearchResult 返回结构中。
    post_process_textual_mem,
)
# 获取项目统一 logger 工厂，保证日志格式和上下文一致。
from memos.log import get_logger
# 解析 LLM 对“新增前过滤”Prompt 的响应，用来判断候选记忆是否保留。
from memos.mem_reader.utils import parse_keep_filter_response
# ScheduleMessageItem 是提交给调度器的任务消息载体。
from memos.mem_scheduler.schemas.message_schemas import ScheduleMessageItem
# 引入调度任务标签，不同标签决定调度器执行新增、读取还是反馈逻辑。
from memos.mem_scheduler.schemas.task_schemas import (
    # ADD_TASK_LABEL 表示同步新增后的后续处理任务。
    ADD_TASK_LABEL,
    # MEM_FEEDBACK_TASK_LABEL 表示反馈记忆处理任务。
    MEM_FEEDBACK_TASK_LABEL,
    # MEM_READ_TASK_LABEL 表示异步读取/抽取记忆任务。
    MEM_READ_TASK_LABEL,
)
# TextualMemoryItem 是文本记忆的核心数据结构，包含 memory 文本和 metadata。
from memos.memories.textual.item import TextualMemoryItem
# MemCubeView 是 cube view 的抽象接口，SingleCubeView 需要实现其中的业务能力。
from memos.multi_mem_cube.views import MemCubeView
# resolve_filter_for_cube 处理多 cube 场景下的过滤条件；search_text_memories 提供快速文本检索入口。
from memos.search import resolve_filter_for_cube, search_text_memories
# PROMPT_MAPPING 存放 Prompt 模板，add_before_search 会取其中的过滤模板。
from memos.templates.mem_reader_prompts import PROMPT_MAPPING
# 引入搜索策略、搜索结果和用户上下文等通用类型。
from memos.types.general_types import (
    # FINE_STRATEGY 控制 fine search 具体采用普通增强、deep search 还是 agentic search。
    FINE_STRATEGY,
    # FineStrategy 枚举定义精细搜索的不同实现策略。
    FineStrategy,
    # MOSSearchResult 是对外返回的多类型记忆搜索结果结构。
    MOSSearchResult,
    # SearchMode 定义 fast、fine、mixture 等搜索模式常量。
    SearchMode,
    # UserContext 统一携带 user、cube、session 等上下文，避免各层重复传散乱参数。
    UserContext,
)
# timed 和 timed_stage 用于方法级、阶段级耗时统计，帮助分析性能瓶颈。
from memos.utils import timed, timed_stage


# 创建当前模块级 logger，供静态函数或没有实例 logger 的位置记录日志。
logger = get_logger(__name__)


# 下面这些导入只服务于类型检查，不会在运行时执行，从而避免循环导入和启动成本。
if TYPE_CHECKING:
    # 类型检查时引入 API 请求模型，避免运行期循环导入。
    from memos.api.product_models import APIADDRequest, APIFeedbackRequest, APISearchRequest
    # 类型检查时引入底层记忆 cube 类型。
    from memos.mem_cube.navie import NaiveMemCube
    # 类型检查时引入记忆读取器类型。
    from memos.mem_reader.simple_struct import SimpleStructMemReader
    # 类型检查时引入调度器类型。
    from memos.mem_scheduler.optimized_scheduler import OptimizedScheduler


# 用 dataclass 声明该视图类，使依赖字段可以通过构造函数直接注入。
@dataclass
# SingleCubeView 表示“单个 memory cube”的统一操作视图，负责 add/search/feedback 等具体流程。
class SingleCubeView(MemCubeView):
    # 当前视图绑定的 memory cube ID，所有读写操作默认限定在这个 cube 中。
    cube_id: str
    # 底层记忆存储对象，真正负责把文本记忆写入或读出。
    naive_mem_cube: NaiveMemCube
    # 记忆读取器负责从对话、文件等输入中抽取结构化记忆。
    mem_reader: SimpleStructMemReader
    # 调度器负责提交异步或后续处理任务，例如记忆读取、融合和反馈处理。
    mem_scheduler: OptimizedScheduler
    # 实例级 logger 由上层传入，保证不同 cube 视图也能输出带上下文的日志。
    logger: Any
    # 搜索组件用于 fast/fine/deep 等检索流程，也用于新增前的相似记忆检查。
    searcher: Any
    # feedback_server 可选注入，只有处理 feedback_memories 时才真正依赖它。
    feedback_server: Any | None = None
    # deepsearch_agent 可选注入，用于 agentic search 策略下的复杂检索。
    deepsearch_agent: Any | None = None

    # timed 装饰器记录整个方法的调用耗时，方便观测接口级性能。
    @timed
    # 处理单个 cube 的新增记忆请求，是 AddHandler 分发到 SingleCubeView 后的主入口。
    def add_memories(self, add_req: APIADDRequest) -> list[dict[str, Any]]:
        """
        This is basically your current handle_add_memories logic,
        but scoped to a single cube_id.
        """
        # 请求中的 async_mode 优先级最高；如果未指定，则从底层 memory cube 读取默认模式。
        sync_mode = add_req.async_mode or self._get_sync_mode()
        # 输出诊断日志，包含 cube、同步模式和完整请求，便于排查新增链路问题。
        self.logger.info(
            # 计算或整理当前步骤所需的中间数据，供后续逻辑使用。
            f"[DIAGNOSTIC] single_cube.add_memories called for cube_id: {self.cube_id}. sync_mode: {sync_mode}. Request: {add_req.model_dump_json(indent=2)}"
        )
        # 构造用户上下文，把用户、cube、session 等身份信息集中传给后续 reader/searcher/scheduler。
        user_context = UserContext(
            # 记录当前请求所属用户。
            user_id=add_req.user_id,
            # 将当前 SingleCubeView 绑定的 cube_id 写入上下文，后续读写都以它作为命名空间。
            mem_cube_id=self.cube_id,
            # 如果请求没有 session_id，则使用默认会话，保证上下文始终可用。
            session_id=add_req.session_id or "default_session",
            # 透传管理者用户 ID，供多用户或代理场景识别上级管理者。
            manager_user_id=add_req.manager_user_id,
            # 透传项目 ID，方便按项目隔离或追踪记忆。
            project_id=add_req.project_id,
        )

        # 再次得到实际 session_id，主要用于日志表达和后续处理的一致性。
        target_session_id = add_req.session_id or "default_session"
        # 输出诊断日志，包含 cube、同步模式和完整请求，便于排查新增链路问题。
        self.logger.info(
            # 计算或整理当前步骤所需的中间数据，供后续逻辑使用。
            f"[SingleCubeView] cube={self.cube_id} "
            # 计算或整理当前步骤所需的中间数据，供后续逻辑使用。
            f"Processing add with mode={sync_mode}, session={target_session_id}"
        )

        # 进入文本记忆处理主流程，完成抽取、写入和调度。
        all_memories = self._process_text_mem(add_req, user_context, sync_mode)

        # 记录日志，帮助观察当前分支的执行状态或异常信息。
        self.logger.info(f"[SingleCubeView] cube={self.cube_id} total_results={len(all_memories)}")

        # 将本 cube 新增成功的记忆结果返回给上层 Handler 或 CompositeCubeView。
        return all_memories

    # timed 装饰器记录整个方法的调用耗时，方便观测接口级性能。
    @timed
    # 处理单个 cube 的搜索请求，将过滤、检索、格式化三个步骤串起来。
    def search_memories(self, search_req: APISearchRequest) -> dict[str, Any]:
        """
        Unified memory search handling (text + preference memories).
        Preference memories are now searched through the same _search_text flow.
        """
        # 将请求中的通用过滤条件解析成适用于当前 cube 的过滤条件。
        cube_filter = resolve_filter_for_cube(search_req.filter, self.cube_id)
        # 如果过滤条件被改写，需要复制请求，避免修改调用方传入的原始对象。
        if cube_filter is not search_req.filter:
            # 局部导入 copy，只在确实需要复制请求对象时才加载。
            import copy

            # 浅拷贝请求模型，使当前 cube 的 filter 调整不会影响其他 cube。
            search_req = copy.copy(search_req)
            # 把当前 cube 专用的过滤条件写回复制后的请求。
            search_req.filter = cube_filter

        # Create UserContext object
        # 构造搜索上下文，确保 searcher 能在正确用户和 cube 范围内检索。
        user_context = UserContext(
            # 计算或整理当前步骤所需的中间数据，供后续逻辑使用。
            user_id=search_req.user_id,
            # 计算或整理当前步骤所需的中间数据，供后续逻辑使用。
            mem_cube_id=self.cube_id,
            # 计算或整理当前步骤所需的中间数据，供后续逻辑使用。
            session_id=search_req.session_id or "default_session",
        )
        # 记录搜索请求，便于复现查询条件。
        self.logger.info(f"Search Req is: {search_req}")

        # 初始化完整搜索结果骨架，即使某些类型没有结果，也保持返回字段稳定。
        memories_result: MOSSearchResult = {
            # 文本记忆结果列表。
            "text_mem": [],
            # 行为记忆结果列表，当前流程默认置空。
            "act_mem": [],
            # 参数记忆结果列表，当前流程默认置空。
            "para_mem": [],
            # 偏好记忆结果列表，会通过文本记忆统一后处理填充。
            "pref_mem": [],
            # 偏好说明字段，默认没有额外说明。
            "pref_note": "",
            # 工具记忆结果列表，当前流程默认置空。
            "tool_mem": [],
            # 技能记忆结果列表，当前流程默认置空。
            "skill_mem": [],
        }

        # Determine search mode
        # 解析实际搜索模式，后续据此选择 fast/fine/mixture。
        search_mode = self._get_search_mode(search_req.mode)

        # Unified search through _search_text (includes all memory types)
        # 统一通过 _search_text 检索文本相关记忆，并得到已经格式化的候选结果。
        all_formatted_memories = self._search_text(search_req, user_context, search_mode)

        # Build result with unified processing
        # 将格式化后的文本记忆归并进统一 MOSSearchResult 结构。
        memories_result = post_process_textual_mem(
            # 传入已有结果骨架，后处理函数会在其基础上填充字段。
            memories_result,
            # 传入搜索得到的所有文本类记忆。
            all_formatted_memories,
            # 标记这些结果来自当前 cube。
            self.cube_id,
        )

        # 输出最终搜索结果，便于调试结果归类和格式化问题。
        self.logger.info(f"Search memories result: {memories_result}")
        # 记录结果结构长度；注意这里是字典键数量，而不一定是实际记忆条数。
        self.logger.info(f"Search {len(memories_result)} memories.")
        # 返回统一结构，供 API 层直接响应。
        return memories_result

    # timed 装饰器记录整个方法的调用耗时，方便观测接口级性能。
    @timed
    # 处理单个 cube 的反馈记忆请求，根据 async_mode 决定提交异步任务或立即处理。
    def feedback_memories(self, feedback_req: APIFeedbackRequest) -> dict[str, Any]:
        # 解析反馈任务所属会话，缺省时使用默认会话保持任务字段完整。
        target_session_id = feedback_req.session_id or "default_session"
        # 异步模式下不立即处理反馈，而是把反馈请求封装成调度任务。
        if feedback_req.async_mode == "async":
            # 将可能失败的外部调用包在 try 中，避免单次异常中断整体流程。
            try:
                # 将 Pydantic 请求转成普通字典后序列化，作为调度任务内容。
                feedback_req_str = json.dumps(feedback_req.model_dump())
                # 构造反馈调度消息，调度器后续会根据 label 执行反馈处理。
                message_item_feedback = ScheduleMessageItem(
                    # 传入用户 ID，定位用户记忆空间。
                    user_id=feedback_req.user_id,
                    # 透传任务 ID，便于调度链路追踪。
                    task_id=feedback_req.task_id,
                    # 计算或整理当前步骤所需的中间数据，供后续逻辑使用。
                    session_id=target_session_id,
                    # 计算或整理当前步骤所需的中间数据，供后续逻辑使用。
                    mem_cube_id=self.cube_id,
                    # 计算或整理当前步骤所需的中间数据，供后续逻辑使用。
                    mem_cube=self.naive_mem_cube,
                    # 使用反馈任务标签，告诉调度器这是 feedback memory 任务。
                    label=MEM_FEEDBACK_TASK_LABEL,
                    # 把完整反馈请求作为任务内容传给调度器。
                    content=feedback_req_str,
                    # 使用 UTC 时间记录任务提交时间，便于跨时区统一排序。
                    timestamp=datetime.utcnow(),
                )
                # Use scheduler submission to ensure tracking and metrics
                # 提交异步反馈任务，并让调度器负责追踪执行和指标。
                self.mem_scheduler.submit_messages(messages=[message_item_feedback])
                # 记录异步反馈任务提交成功。
                self.logger.info(f"[SingleCubeView] cube={self.cube_id} Submitted FEEDBACK async")
            # 兜底捕获调度提交异常，避免异步任务提交失败直接击穿接口。
            except Exception as e:
                # 记录调度提交失败，并带上异常堆栈方便排查。
                self.logger.error(
                    # 计算或整理当前步骤所需的中间数据，供后续逻辑使用。
                    f"[SingleCubeView] cube={self.cube_id} Failed to submit FEEDBACK: {e}",
                    # 计算或整理当前步骤所需的中间数据，供后续逻辑使用。
                    exc_info=True,
                )
            # 异步模式只表示任务已提交或尝试提交，不返回具体反馈处理结果。
            return []
        # 当前分支处理前面条件不成立时的默认路径。
        else:
            # 同步模式下直接调用反馈服务处理反馈内容。
            feedback_result = self.feedback_server.process_feedback(
                # 传入用户 ID，定位用户记忆空间。
                user_id=feedback_req.user_id,
                # feedback_server 使用 user_name 表示 cube 命名空间，这里传当前 cube_id。
                user_name=self.cube_id,
                # 透传原始 session_id，让反馈处理可关联会话。
                session_id=feedback_req.session_id,
                # 传入反馈发生前的对话历史，帮助服务理解上下文。
                chat_history=feedback_req.history,
                # 如果反馈针对已检索记忆，透传相关记忆 ID。
                retrieved_memory_ids=feedback_req.retrieved_memory_ids,
                # 传入用户反馈文本，这是反馈处理的核心输入。
                feedback_content=feedback_req.feedback_content,
                # 透传反馈时间，便于记录和排序。
                feedback_time=feedback_req.feedback_time,
                # 透传同步/异步模式，保持处理服务上下文完整。
                async_mode=feedback_req.async_mode,
                # 如果用户提供了修正答案，也一起交给反馈服务。
                corrected_answer=feedback_req.corrected_answer,
                # 透传任务 ID，便于调度链路追踪。
                task_id=feedback_req.task_id,
                # 透传附加元信息，用于定制化处理或审计。
                info=feedback_req.info,
            )
            # 记录同步反馈处理结果。
            self.logger.info(f"[Feedback memories result:] {feedback_result}")
        # 返回同步反馈服务给出的处理结果。
        return feedback_result

    # 返回本次搜索使用的模式；当前实现直接信任请求中的 mode。
    def _get_search_mode(self, mode: str) -> str:
        """
        Get search mode with environment variable fallback.

        Args:
            mode: Requested search mode

        Returns:
            Search mode string
        """
        # 当前没有额外配置覆盖逻辑，直接返回请求指定的搜索模式。
        return mode

    # timed 装饰器记录整个方法的调用耗时，方便观测接口级性能。
    @timed
    # 根据搜索模式分派到 fast、fine 或 mixture 搜索流程。
    def _search_text(
        self,
        search_req: APISearchRequest,
        user_context: UserContext,
        search_mode: str,
    ) -> list[dict[str, Any]]:
        """
        Search text memories based on mode.

        Args:
            search_req: Search request
            user_context: User context
            search_mode: Search mode (fast, fine, or mixture)

        Returns:
            List of formatted memory items
        """
        # 将可能失败的外部调用包在 try 中，避免单次异常中断整体流程。
        try:
            # fast 模式走轻量检索路径，优先保证速度。
            if search_mode == SearchMode.FAST:
                # 调用 fast search 获取文本记忆结果。
                text_memories = self._fast_search(search_req, user_context)
            # fine 模式走精细检索路径，可能包含增强、补召回和去重。
            elif search_mode == SearchMode.FINE:
                # 调用 fine search 获取更高质量的文本记忆结果。
                text_memories = self._fine_search(search_req, user_context)
            # mixture 模式组合 fast 和 fine 的优点。
            elif search_mode == SearchMode.MIXTURE:
                # 将混合检索交给调度器封装的 mix_search_memories。
                text_memories = self._mix_search(search_req, user_context)
            # 当前分支处理前面条件不成立时的默认路径。
            else:
                # 对未知搜索模式记录错误，避免静默返回错误结果。
                self.logger.error(f"Unsupported search mode: {search_mode}")
                # 不支持的模式或异常场景返回空列表，保持接口稳定。
                return []
            # 将对应搜索路径得到的结果返回给 search_memories。
            return text_memories

        # 捕获搜索链路中的任何异常，防止单次搜索导致服务崩溃。
        except Exception as e:
            # 输出异常和完整 traceback，便于定位失败发生在哪个搜索阶段。
            self.logger.error("Error in search_text: %s; traceback: %s", e, traceback.format_exc())
            # 不支持的模式或异常场景返回空列表，保持接口稳定。
            return []

    # deep search 路径会调用 searcher.deep_search，通常用于更强的召回和互联网增强场景。
    def _deep_search(
        self,
        search_req: APISearchRequest,
        user_context: UserContext,
    ) -> list:
        # 解析本次深度搜索所属会话，缺省时使用默认会话。
        target_session_id = search_req.session_id or "default_session"
        # 如果请求指定 session，则在深度搜索中优先限制到该会话范围。
        search_filter = {"session_id": search_req.session_id} if search_req.session_id else None

        # 组装透传给搜索组件的上下文信息。
        info = {
            # 传入用户 ID，供搜索组件做审计或个性化处理。
            "user_id": search_req.user_id,
            # 传入实际使用的 session_id。
            "session_id": target_session_id,
            # 传入对话历史，帮助深度搜索理解查询上下文。
            "chat_history": search_req.chat_history,
        }

        # 调用深度搜索接口，它可能结合更复杂的召回、重排或联网能力。
        enhanced_memories = self.searcher.deep_search(
            # 本次搜索的用户查询文本。
            query=search_req.query,
            # 指定搜索命名空间为当前 cube。
            user_name=user_context.mem_cube_id,
            # 限制返回候选数量。
            top_k=search_req.top_k,
            # 深度搜索内部仍按 fine 模式进行高质量检索。
            mode=SearchMode.FINE,
            # 根据请求决定是否关闭互联网搜索能力。
            manual_close_internet=not search_req.internet_search,
            # 透传 moscube 参数，支持搜索器内部的 cube 相关扩展。
            moscube=search_req.moscube,
            # 应用会话过滤条件。
            search_filter=search_filter,
            # 传入额外上下文。
            info=info,
        )
        # 深度搜索结果仍通过统一后格式化函数收口。
        return self._postformat_memories(
            enhanced_memories,
            user_context.mem_cube_id,
            # 只有相似度去重需要 embedding 时才把 embedding 一起返回。
            include_embedding=search_req.dedup == "sim",
            # 根据请求决定是否补充 RawFileMemory 的前后邻居。
            neighbor_discovery=search_req.neighbor_discovery,
        )

    # agentic search 路径交给 deepsearch_agent 自主规划检索，再统一格式化结果。
    def _agentic_search(
        self, search_req: APISearchRequest, user_context: UserContext, max_thinking_depth: int
    ) -> list:
        # 调用 agent，让它根据查询自主执行更复杂的检索计划。
        deepsearch_results = self.deepsearch_agent.run(
            # 传入查询和 cube 命名空间，让 agent 在正确范围内工作。
            search_req.query, user_id=user_context.mem_cube_id
        )
        # agentic 搜索结果也复用统一格式化逻辑。
        return self._postformat_memories(
            deepsearch_results,
            user_context.mem_cube_id,
            # 计算或整理当前步骤所需的中间数据，供后续逻辑使用。
            include_embedding=search_req.dedup == "sim",
            # 计算或整理当前步骤所需的中间数据，供后续逻辑使用。
            neighbor_discovery=search_req.neighbor_discovery,
        )

    # fine search 是精细检索流程，会先召回，再重排/增强，并在必要时补召回。
    def _fine_search(
        self,
        search_req: APISearchRequest,
        user_context: UserContext,
    ) -> list:
        """
        Fine-grained search with query enhancement.

        Args:
            search_req: Search request
            user_context: User context

        Returns:
            List of enhanced search results
        """
        # TODO: support tool memory search in future

        # 记录当前 fine search 策略，便于判断实际走的是哪条检索路径。
        logger.info(f"Fine strategy: {FINE_STRATEGY}")
        # 如果全局策略指定 deep search，则直接切换到深度搜索实现。
        if FINE_STRATEGY == FineStrategy.DEEP_SEARCH:
            # 交给深度搜索处理并返回结果。
            return self._deep_search(search_req=search_req, user_context=user_context)
        # 如果策略指定 agentic search，则使用 agent 驱动的检索实现。
        elif FINE_STRATEGY == FineStrategy.AGENTIC_SEARCH:
            # 交给 agentic 搜索处理并返回结果。
            return self._agentic_search(search_req=search_req, user_context=user_context)

        # 普通 fine search 先解析会话 ID。
        target_session_id = search_req.session_id or "default_session"
        # 有 session_id 时将其作为搜索优先级，帮助检索更贴近当前会话。
        search_priority = {"session_id": search_req.session_id} if search_req.session_id else None
        # 使用请求中传入的过滤条件，前面 search_memories 已经按 cube 做过适配。
        search_filter = search_req.filter

        # 组装传给 searcher 和 retriever 的上下文信息。
        info = {
            "user_id": search_req.user_id,
            "session_id": target_session_id,
            "chat_history": search_req.chat_history,
        }

        # Fine retrieve
        # 第一步精细召回：调用 searcher.retrieve 获取原始候选。
        raw_retrieved_memories = self.searcher.retrieve(
            # 计算或整理当前步骤所需的中间数据，供后续逻辑使用。
            query=search_req.query,
            # 计算或整理当前步骤所需的中间数据，供后续逻辑使用。
            user_name=user_context.mem_cube_id,
            # 计算或整理当前步骤所需的中间数据，供后续逻辑使用。
            top_k=search_req.top_k,
            # 计算或整理当前步骤所需的中间数据，供后续逻辑使用。
            mode=SearchMode.FINE,
            # 指定要检索的记忆类型。
            memory_type=search_req.search_memory_type,
            # 计算或整理当前步骤所需的中间数据，供后续逻辑使用。
            manual_close_internet=not search_req.internet_search,
            # 计算或整理当前步骤所需的中间数据，供后续逻辑使用。
            moscube=search_req.moscube,
            # 计算或整理当前步骤所需的中间数据，供后续逻辑使用。
            search_filter=search_filter,
            # 传入会话优先级，使召回更关注当前上下文。
            search_priority=search_priority,
            # 计算或整理当前步骤所需的中间数据，供后续逻辑使用。
            info=info,
        )

        # Post retrieve
        # 第二步后处理召回：对原始候选进行重排、裁剪或去重。
        raw_memories = self.searcher.post_retrieve(
            # 传入第一步召回结果。
            retrieved_results=raw_retrieved_memories,
            # 计算或整理当前步骤所需的中间数据，供后续逻辑使用。
            top_k=search_req.top_k,
            # 计算或整理当前步骤所需的中间数据，供后续逻辑使用。
            user_name=user_context.mem_cube_id,
            # 计算或整理当前步骤所需的中间数据，供后续逻辑使用。
            info=info,
            # 按请求控制后处理阶段的去重策略。
            dedup=search_req.dedup,
        )

        # Enhance with query
        # 第三步查询增强：让 retriever 判断哪些记忆真正与当前 query 相关。
        enhanced_memories, _ = self.mem_scheduler.retriever.enhance_memories_with_query(
            # 当前只使用本次 query 作为查询历史。
            query_history=[search_req.query],
            # 传入后处理后的候选记忆。
            memories=raw_memories,
        )

        # 如果增强过滤后记忆数量减少，说明有候选被剔除，可能需要补召回。
        if len(enhanced_memories) < len(raw_memories):
            # 记录日志，帮助观察当前分支的执行状态或异常信息。
            logger.info(
                f"Enhanced memories ({len(enhanced_memories)}) are less than raw memories ({len(raw_memories)}). Recalling for more."
            )
            # 让 retriever 判断是否存在信息缺口，以及是否需要用 hint 重新召回。
            missing_info_hint, trigger = self.mem_scheduler.retriever.recall_for_missing_memories(
                # 计算或整理当前步骤所需的中间数据，供后续逻辑使用。
                query=search_req.query,
                # 计算或整理当前步骤所需的中间数据，供后续逻辑使用。
                memories=[mem.memory for mem in enhanced_memories],
            )
            # 计算需要补充的候选数量，使最终数量尽量接近原召回规模。
            retrieval_size = len(raw_memories) - len(enhanced_memories)
            # 记录日志，帮助观察当前分支的执行状态或异常信息。
            logger.info(f"Retrieval size: {retrieval_size}")
            # trigger 为真时使用缺失信息提示词进行额外搜索。
            if trigger:
                # 记录日志，帮助观察当前分支的执行状态或异常信息。
                logger.info(f"Triggering additional search with hint: {missing_info_hint}")
                # 根据 missing_info_hint 发起一次额外 fast search，用来补齐缺失信息。
                additional_memories = self.searcher.search(
                    # 额外搜索不再用原 query，而是用 retriever 生成的缺口提示。
                    query=missing_info_hint,
                    # 计算或整理当前步骤所需的中间数据，供后续逻辑使用。
                    user_name=user_context.mem_cube_id,
                    # 计算或整理当前步骤所需的中间数据，供后续逻辑使用。
                    top_k=retrieval_size,
                    # 补召回使用 fast 模式，降低二次检索成本。
                    mode=SearchMode.FAST,
                    # 指定要检索的记忆类型。
                    memory_type=search_req.search_memory_type,
                    # 传入会话优先级，使召回更关注当前上下文。
                    search_priority=search_priority,
                    # 计算或整理当前步骤所需的中间数据，供后续逻辑使用。
                    search_filter=search_filter,
                    # 计算或整理当前步骤所需的中间数据，供后续逻辑使用。
                    info=info,
                )
            # 当前分支处理前面条件不成立时的默认路径。
            else:
                # 记录日志，帮助观察当前分支的执行状态或异常信息。
                logger.info("Not triggering additional search, using fast memories.")
                # 从原始候选中截取缺口数量，保证结果数量不至于过少。
                additional_memories = raw_memories[:retrieval_size]

            # 将补充候选追加到增强结果中。
            enhanced_memories += additional_memories
            # 记录日志，帮助观察当前分支的执行状态或异常信息。
            logger.info(
                f"Added {len(additional_memories)} more memories. Total enhanced memories: {len(enhanced_memories)}"
            )

        # 定义局部去重函数，按 memory 文本规范化后的内容去除重复项。
        def _dedup_by_content(memories: list) -> list:
            # 计算或整理当前步骤所需的中间数据，供后续逻辑使用。
            seen = set()
            # 计算或整理当前步骤所需的中间数据，供后续逻辑使用。
            unique_memories = []
            # 遍历当前集合，逐项处理并累积结果。
            for mem in memories:
                # 计算或整理当前步骤所需的中间数据，供后续逻辑使用。
                key = " ".join(mem.memory.split())
                # 根据当前状态或请求参数选择不同处理分支。
                if key in seen:
                    continue
                seen.add(key)
                unique_memories.append(mem)
            # 返回当前流程计算出的结果，交给上层调用方继续处理。
            return unique_memories

        # 计算或整理当前步骤所需的中间数据，供后续逻辑使用。
        deduped_memories = (
            # 计算或整理当前步骤所需的中间数据，供后续逻辑使用。
            enhanced_memories if search_req.dedup == "no" else _dedup_by_content(enhanced_memories)
        )
        # 计算或整理当前步骤所需的中间数据，供后续逻辑使用。
        formatted_memories = self._postformat_memories(
            deduped_memories,
            user_context.mem_cube_id,
            # 计算或整理当前步骤所需的中间数据，供后续逻辑使用。
            include_embedding=search_req.dedup == "sim",
            # 计算或整理当前步骤所需的中间数据，供后续逻辑使用。
            neighbor_discovery=search_req.neighbor_discovery,
        )

        # 记录日志，帮助观察当前分支的执行状态或异常信息。
        logger.info(f"Found {len(formatted_memories)} memories for user {search_req.user_id}")

        # 返回当前流程计算出的结果，交给上层调用方继续处理。
        return formatted_memories

    # fast search 直接走向量/文本检索，重点是速度和基础召回。
    def _fast_search(
        self,
        search_req: APISearchRequest,
        user_context: UserContext,
    ) -> list:
        """
        Fast search using vector database.

        Args:
            search_req: Search request
            user_context: User context

        Returns:
            List of search results
        """
        # 调用统一的文本记忆搜索工具执行快速检索。
        search_results = search_text_memories(
            # 指定底层文本记忆存储作为检索数据源。
            text_mem=self.naive_mem_cube.text_mem,
            # 传入原始搜索请求，包含 query/top_k/filter 等条件。
            search_req=search_req,
            # 传入用户上下文，保证在正确用户和 cube 范围内检索。
            user_context=user_context,
            # 明确使用 fast 模式。
            mode=SearchMode.FAST,
            # MMR 或相似度去重需要 embedding，因此按需附带。
            include_embedding=(search_req.dedup in ("mmr", "sim")),
        )

        # 快速检索结果同样走统一格式化出口。
        return self._postformat_memories(
            search_results,
            user_context.mem_cube_id,
            # MMR 或相似度去重需要 embedding，因此按需附带。
            include_embedding=(search_req.dedup in ("mmr", "sim")),
            # 计算或整理当前步骤所需的中间数据，供后续逻辑使用。
            neighbor_discovery=search_req.neighbor_discovery,
        )

    # 对搜索结果做邻居扩展和统一格式化，是所有搜索路径的收口函数。
    def _postformat_memories(
        self,
        search_results: list,
        user_name: str,
        # 计算或整理当前步骤所需的中间数据，供后续逻辑使用。
        include_embedding: bool = False,
        # 计算或整理当前步骤所需的中间数据，供后续逻辑使用。
        neighbor_discovery: bool = False,
    ) -> list:
        """
        Postprocess search results.
        """

        # 从图边信息中取相邻记忆节点，用于 RawFileMemory 的上下文邻居补全。
        def extract_edge_info(edges_info: list[dict], neighbor_relativity: float):
            # 计算或整理当前步骤所需的中间数据，供后续逻辑使用。
            edge_mems = []
            # 遍历当前集合，逐项处理并累积结果。
            for edge in edges_info:
                # 计算或整理当前步骤所需的中间数据，供后续逻辑使用。
                chunk_target_id = edge.get("to")
                # 计算或整理当前步骤所需的中间数据，供后续逻辑使用。
                edge_type = edge.get("type")
                # 计算或整理当前步骤所需的中间数据，供后续逻辑使用。
                item_neighbor = self.searcher.graph_store.get_node(
                    # 计算或整理当前步骤所需的中间数据，供后续逻辑使用。
                    chunk_target_id, user_name=user_name
                )
                # 根据当前状态或请求参数选择不同处理分支。
                if item_neighbor:
                    # 计算或整理当前步骤所需的中间数据，供后续逻辑使用。
                    item_neighbor_mem = TextualMemoryItem(**item_neighbor)
                    # 计算或整理当前步骤所需的中间数据，供后续逻辑使用。
                    item_neighbor_mem.metadata.relativity = neighbor_relativity
                    edge_mems.append(item_neighbor_mem)
                    # 计算或整理当前步骤所需的中间数据，供后续逻辑使用。
                    item_neighbor_id = item_neighbor.get("id", "None")
                    # 记录日志，帮助观察当前分支的执行状态或异常信息。
                    self.logger.info(
                        f"Add neighbor chunk: {item_neighbor_id}, edge_type: {edge_type} for {item.id}"
                    )
            # 返回当前流程计算出的结果，交给上层调用方继续处理。
            return edge_mems

        # 计算或整理当前步骤所需的中间数据，供后续逻辑使用。
        final_items = []
        # 根据当前状态或请求参数选择不同处理分支。
        if neighbor_discovery:
            # 遍历当前集合，逐项处理并累积结果。
            for item in search_results:
                # 根据当前状态或请求参数选择不同处理分支。
                if item.metadata.memory_type == "RawFileMemory":
                    # 计算或整理当前步骤所需的中间数据，供后续逻辑使用。
                    neighbor_relativity = item.metadata.relativity * 0.8
                    # 计算或整理当前步骤所需的中间数据，供后续逻辑使用。
                    preceding_info = self.searcher.graph_store.get_edges(
                        # 计算或整理当前步骤所需的中间数据，供后续逻辑使用。
                        item.id, type="PRECEDING", direction="OUTGOING", user_name=user_name
                    )
                    final_items.extend(extract_edge_info(preceding_info, neighbor_relativity))

                    final_items.append(item)

                    # 计算或整理当前步骤所需的中间数据，供后续逻辑使用。
                    following_info = self.searcher.graph_store.get_edges(
                        # 计算或整理当前步骤所需的中间数据，供后续逻辑使用。
                        item.id, type="FOLLOWING", direction="OUTGOING", user_name=user_name
                    )
                    final_items.extend(extract_edge_info(following_info, neighbor_relativity))

                # 当前分支处理前面条件不成立时的默认路径。
                else:
                    final_items.append(item)
        # 当前分支处理前面条件不成立时的默认路径。
        else:
            # 计算或整理当前步骤所需的中间数据，供后续逻辑使用。
            final_items = search_results

        # 返回当前流程计算出的结果，交给上层调用方继续处理。
        return [
            # 计算或整理当前步骤所需的中间数据，供后续逻辑使用。
            format_memory_item(data, include_embedding=include_embedding) for data in final_items
        ]

    # mixture search 组合 fast 与 fine 的能力，具体策略交给调度器实现。
    def _mix_search(
        self,
        search_req: APISearchRequest,
        user_context: UserContext,
    ) -> list:
        """
        Mix search combining fast and fine-grained approaches.

        Args:
            search_req: Search request
            user_context: User context

        Returns:
            List of formatted search results
        """
        # 混合搜索具体策略由调度器实现，这里只负责转发请求和上下文。
        return self.mem_scheduler.mix_search_memories(
            # 计算或整理当前步骤所需的中间数据，供后续逻辑使用。
            search_req=search_req,
            # 计算或整理当前步骤所需的中间数据，供后续逻辑使用。
            user_context=user_context,
        )

    # 从底层 text_mem 获取默认同步模式，拿不到时回退为 sync。
    def _get_sync_mode(self) -> str:
        """
        Get synchronization mode from memory cube.

        Returns:
            Sync mode string ("sync" or "async")
        """
        # 将可能失败的外部调用包在 try 中，避免单次异常中断整体流程。
        try:
            # 优先读取 text_mem.mode；如果没有该属性，就默认 sync。
            return getattr(self.naive_mem_cube.text_mem, "mode", "sync")
        # 任何读取异常都回退默认值，保证新增流程不会因此中断。
        except Exception:
            # 默认采用同步模式，语义更保守，也便于调用方立即拿到结果。
            return "sync"

    # 根据同步模式把后续记忆处理任务提交给调度器。
    def _schedule_memory_tasks(
        self,
        add_req: APIADDRequest,
        user_context: UserContext,
        mem_ids: list[str],
        sync_mode: str,
    ) -> None:
        """
        Schedule memory processing tasks based on sync mode.

        Args:
            add_req: Add memory request
            user_context: User context
            mem_ids: List of memory IDs
            sync_mode: Synchronization mode
        """
        # 解析任务所属会话，缺省时使用默认会话。
        target_session_id = add_req.session_id or "default_session"

        # 异步模式下提交 MEM_READ 任务，让后台继续处理记忆读取和后续流程。
        if sync_mode == "async":
            # Async mode: submit MEM_READ_LABEL task
            # 将可能失败的外部调用包在 try 中，避免单次异常中断整体流程。
            try:
                # 构造异步读取任务消息。
                message_item_read = ScheduleMessageItem(
                    # 计算或整理当前步骤所需的中间数据，供后续逻辑使用。
                    user_id=add_req.user_id,
                    # 计算或整理当前步骤所需的中间数据，供后续逻辑使用。
                    task_id=add_req.task_id,
                    # 计算或整理当前步骤所需的中间数据，供后续逻辑使用。
                    session_id=target_session_id,
                    # 计算或整理当前步骤所需的中间数据，供后续逻辑使用。
                    mem_cube_id=self.cube_id,
                    # 计算或整理当前步骤所需的中间数据，供后续逻辑使用。
                    mem_cube=self.naive_mem_cube,
                    # 使用 MEM_READ 标签，调度器会把它识别为记忆读取任务。
                    label=MEM_READ_TASK_LABEL,
                    # 将新写入的记忆 ID 列表序列化，作为后续任务处理对象。
                    content=json.dumps(mem_ids),
                    # 计算或整理当前步骤所需的中间数据，供后续逻辑使用。
                    timestamp=datetime.utcnow(),
                    # 指定任务执行时使用的 cube 命名空间。
                    user_name=self.cube_id,
                    # 在调度任务中附带额外元信息，供后台处理使用。
                    info={
                        # 先展开请求中已有的 info；为空时使用空字典避免报错。
                        **(add_req.info or {}),
                        # 额外标记是否为技能上传场景，缺省为 False。
                        "is_upload_skill": getattr(add_req, "is_upload_skill", False),
                    },
                    # 把聊天历史交给后续任务，便于上下文处理。
                    chat_history=add_req.chat_history,
                    # 传入完整用户上下文，后台任务无需重新组装。
                    user_context=user_context,
                )
                # 把异步读取任务提交给调度器。
                self.mem_scheduler.submit_messages(messages=[message_item_read])
                # 记录日志，帮助观察当前分支的执行状态或异常信息。
                self.logger.info(
                    # 计算或整理当前步骤所需的中间数据，供后续逻辑使用。
                    f"[SingleCubeView] cube={self.cube_id} Submitted async MEM_READ: {json.dumps(mem_ids)}"
                )
            # 捕获异步任务提交失败，避免新增主流程被异常中断。
            except Exception as e:
                # 记录日志，帮助观察当前分支的执行状态或异常信息。
                self.logger.error(
                    # 计算或整理当前步骤所需的中间数据，供后续逻辑使用。
                    f"[SingleCubeView] cube={self.cube_id} Failed to submit async memory tasks: {e}",
                    # 计算或整理当前步骤所需的中间数据，供后续逻辑使用。
                    exc_info=True,
                )
        # 当前分支处理前面条件不成立时的默认路径。
        else:
            # 同步模式下也提交 ADD 任务，用于后续调度链路处理。
            message_item_add = ScheduleMessageItem(
                # 计算或整理当前步骤所需的中间数据，供后续逻辑使用。
                user_id=add_req.user_id,
                # 计算或整理当前步骤所需的中间数据，供后续逻辑使用。
                task_id=add_req.task_id,
                # 计算或整理当前步骤所需的中间数据，供后续逻辑使用。
                session_id=target_session_id,
                # 计算或整理当前步骤所需的中间数据，供后续逻辑使用。
                mem_cube_id=self.cube_id,
                # 计算或整理当前步骤所需的中间数据，供后续逻辑使用。
                mem_cube=self.naive_mem_cube,
                # 使用 ADD 标签，表示这是新增记忆后的处理任务。
                label=ADD_TASK_LABEL,
                # 将新写入的记忆 ID 列表序列化，作为后续任务处理对象。
                content=json.dumps(mem_ids),
                # 计算或整理当前步骤所需的中间数据，供后续逻辑使用。
                timestamp=datetime.utcnow(),
                # 指定任务执行时使用的 cube 命名空间。
                user_name=self.cube_id,
            )
            # 提交同步新增后的调度任务。
            self.mem_scheduler.submit_messages(messages=[message_item_add])

    # 在正式写入前检索相关旧记忆，并借助 LLM 判断新记忆是否应该保留。
    def add_before_search(
        self,
        messages: list[dict],
        memory_list: list[TextualMemoryItem],
        user_name: str,
        info: dict[str, Any],
    ) -> list[TextualMemoryItem]:
        # Build input objects with memory text and metadata (timestamps, sources, etc.)
        # 取新增前过滤的 Prompt 模板，用于让 LLM 判断候选记忆是否值得保留。
        template = PROMPT_MAPPING["add_before_search"]

        # 如果没有搜索器，就无法查找相关旧记忆，只能跳过过滤。
        if not self.searcher:
            # 记录搜索器未初始化的告警，方便发现配置问题。
            self.logger.warning("[add_before_search] Searcher is not initialized, skipping check.")
            # 解析失败时保守保留全部候选，避免误删。
            return memory_list

        # 1. Gather candidates and search for related memories
        # 用于保存每条新记忆及其相关旧记忆，后续会拼进 Prompt。
        candidates_data = []
        # 遍历候选记忆，同时保留原始下标，方便 LLM 响应映射回 memory_list。
        for idx, mem in enumerate(memory_list):
            # 将可能失败的外部调用包在 try 中，避免单次异常中断整体流程。
            try:
                # 用新记忆文本去搜索相似旧记忆，帮助判断它是否重复或冲突。
                related_memories = self.searcher.search(
                    # 每条候选只取少量快速相关结果，控制过滤阶段成本。
                    query=mem.memory, top_k=3, mode="fast", user_name=user_name, info=info
                )
                # 默认认为没有相关旧记忆。
                related_text = "None"
                # 如果检索到了相关记忆，则把它们整理成文本。
                if related_memories:
                    # 将相关记忆按项目符号拼接，便于放进 Prompt 给 LLM 阅读。
                    related_text = "\n".join([f"- {r.memory}" for r in related_memories])

                # 把当前候选及其相关记忆保存下来。
                candidates_data.append(
                    # 记录候选下标、新记忆文本和相关旧记忆文本。
                    {"idx": idx, "new_memory": mem.memory, "related_memories": related_text}
                )
            # 单条候选搜索失败时只影响该候选，不终止整个过滤流程。
            except Exception as e:
                # 记录日志，帮助观察当前分支的执行状态或异常信息。
                self.logger.error(
                    f"[add_before_search] Search error for memory '{mem.memory}': {e}"
                )
                # If search fails, we can either skip this check or treat related as empty
                # 把当前候选及其相关记忆保存下来。
                candidates_data.append(
                    {
                        "idx": idx,
                        "new_memory": mem.memory,
                        # 标记该候选的相似搜索失败，让 Prompt 仍能继续构造。
                        "related_memories": "None (Search Failed)",
                    }
                )

        # 没有候选数据时，无需调用 LLM 过滤。
        if not candidates_data:
            # 解析失败时保守保留全部候选，避免误删。
            return memory_list

        # 2. Build Prompt
        # 将原始消息列表展开成可读文本，作为 LLM 判断记忆价值的上下文。
        messages_inline = "\n".join(
            [
                # 每条消息按 role 和 content 格式化，缺失字段时给默认值。
                f"- [{message.get('role', 'unknown')}]: {message.get('content', '')}"
                # 遍历本次新增记忆来源的对话消息。
                for message in messages
            ]
        )

        # 将候选记忆整理成以字符串下标为 key 的字典，方便 LLM 按索引返回判断。
        candidates_inline_dict = {
            # 用字符串形式的 idx 作为 JSON key，避免 LLM 输出时混淆。
            str(item["idx"]): {
                # 写入候选新记忆文本。
                "new_memory": item["new_memory"],
                # 写入该候选对应的相关旧记忆。
                "related_memories": item["related_memories"],
            }
            # 遍历前面收集的所有候选数据。
            for item in candidates_data
        }

        # 将候选字典转成格式化 JSON；ensure_ascii=False 保留中文可读性。
        candidates_inline = json.dumps(candidates_inline_dict, ensure_ascii=False, indent=2)

        # 把消息上下文和候选记忆填入 Prompt 模板。
        prompt = template.format(
            # 同时传入对话内容和候选内容，让 LLM 结合上下文判断。
            messages_inline=messages_inline, candidates_inline=candidates_inline
        )

        # 3. Call LLM
        # 将可能失败的外部调用包在 try 中，避免单次异常中断整体流程。
        try:
            # 调用通用 LLM，让它根据 Prompt 输出每条候选的 keep/filter 判断。
            raw = self.mem_reader.general_llm.generate([{"role": "user", "content": prompt}])
            # 解析 LLM 原始输出，得到是否成功和结构化过滤结果。
            success, parsed_result = parse_keep_filter_response(raw)

            # 如果 LLM 输出无法解析，不能信任过滤结果。
            if not success:
                # 记录日志，帮助观察当前分支的执行状态或异常信息。
                self.logger.warning(
                    "[add_before_search] Failed to parse LLM response, keeping all."
                )
                # 解析失败时保守保留全部候选，避免误删。
                return memory_list

            # 4. Filter
            # 保存最终保留下来的记忆。
            filtered_list = []
            # 遍历候选记忆，同时保留原始下标，方便 LLM 响应映射回 memory_list。
            for idx, mem in enumerate(memory_list):
                # 取当前候选对应的 LLM 判断结果。
                res = parsed_result.get(idx)
                # 如果 LLM 没有给当前 idx 的结果，则保守保留。
                if not res:
                    # 当前候选被判定保留，加入过滤后的列表。
                    filtered_list.append(mem)
                    continue

                # 默认 keep=True，只有明确要求丢弃时才过滤。
                if res.get("keep", True):
                    # 当前候选被判定保留，加入过滤后的列表。
                    filtered_list.append(mem)
                # 当前分支处理前面条件不成立时的默认路径。
                else:
                    # 记录日志，帮助观察当前分支的执行状态或异常信息。
                    self.logger.info(
                        f"[add_before_search] Dropping memory: '{mem.memory}', reason: '{res.get('reason')}'"
                    )

            # 返回经过 LLM 过滤后的候选记忆列表。
            return filtered_list

        # 单条候选搜索失败时只影响该候选，不终止整个过滤流程。
        except Exception as e:
            # LLM 调用失败时记录错误。
            self.logger.error(f"[add_before_search] LLM execution error: {e}")
            # 解析失败时保守保留全部候选，避免误删。
            return memory_list

    # timed 装饰器记录整个方法的调用耗时，方便观测接口级性能。
    @timed
    # 处理文本记忆新增的核心流程：抽取、写库、调度、归档和格式化返回。
    def _process_text_mem(
        self,
        add_req: APIADDRequest,
        user_context: UserContext,
        sync_mode: str,
    ) -> list[dict[str, Any]]:
        """
        Process and add text memories (including preference memories).

        Extracts memories from messages and adds them to the text memory system.
        Handles both sync and async modes.

        Args:
            add_req: Add memory request
            user_context: User context with IDs

        Returns:
            List of formatted memory responses
        """
        # 解析本次新增所属会话，缺省时使用默认会话。
        target_session_id = add_req.session_id or "default_session"

        # Decide extraction mode:
        # - async: always fast (ignore add_req.mode)
        # - sync: use add_req.mode == "fast" to switch to fast pipeline, otherwise fine
        # 异步模式下优先降低请求耗时，因此固定使用 fast 抽取。
        if sync_mode == "async":
            # fast 抽取更轻量，适合异步入口快速返回。
            extract_mode = "fast"
        # 同步模式下可以根据请求显式选择 fast，否则默认 fine 以保证质量。
        else:  # sync
            # 同步请求若 mode=fast 则走快速抽取，否则走精细抽取。
            extract_mode = "fast" if add_req.mode == "fast" else "fine"

        # 记录日志，帮助观察当前分支的执行状态或异常信息。
        self.logger.info(
            # 计算或整理当前步骤所需的中间数据，供后续逻辑使用。
            "[SingleCubeView] cube=%s Processing text memory "
            # 计算或整理当前步骤所需的中间数据，供后续逻辑使用。
            "with sync_mode=%s, extract_mode=%s, add_mode=%s",
            user_context.mem_cube_id,
            sync_mode,
            extract_mode,
            add_req.mode,
        )
        # 记录整个文本记忆处理开始时间，用于最后计算总耗时。
        process_start = time.perf_counter()

        # Stage 1+2: parse + embedding (logged inside get_memory via timed_stage)
        # 开始 get_memory 阶段计时，覆盖记忆抽取和可能的 embedding 过程。
        with timed_stage("add", "get_memory", cube_id=self.cube_id) as ts_gm:
            # 调用 mem_reader 从消息中抽取候选记忆。
            memories_local = self.mem_reader.get_memory(
                # get_memory 接收批量输入，因此把本次 messages 包成一层列表。
                [add_req.messages],
                # 指定输入类型为聊天记录。
                type="chat",
                # 给抽取器传入上下文信息和用户自定义元信息。
                info={
                    # 传入当前调用所需的参数，保持上下文和配置向下游传递。
                    **(add_req.info or {}),
                    # 传入自定义标签，使抽取出的记忆可带上用户标签。
                    "custom_tags": add_req.custom_tags,
                    # 把用户 ID 放进 info，方便抽取器或下游记录。
                    "user_id": add_req.user_id,
                    # 把实际会话 ID 放进 info。
                    "session_id": target_session_id,
                },
                # 使用前面决定的抽取模式。
                mode=extract_mode,
                # 指定写入命名空间，确保记忆进入当前 cube。
                user_name=user_context.mem_cube_id,
                # 传入历史对话，帮助抽取器理解当前消息上下文。
                chat_history=add_req.chat_history,
                # 传入完整用户上下文，避免抽取器重复组装。
                user_context=user_context,
                # 标记是否为技能上传场景，缺省为 False。
                is_upload_skill=getattr(add_req, "is_upload_skill", False),
            )
        # 读取 get_memory 阶段耗时，后续汇总到 summary。
        get_memory_ms = ts_gm.duration_ms
        # get_memory 返回批次嵌套列表，这里拉平成单层记忆列表。
        flattened_local = [mm for m in memories_local for mm in m]

        # Explicitly set source_doc_id to metadata if present in info
        # 如果 info 中指定了来源文档 ID，就取出来写入每条记忆 metadata。
        source_doc_id = (add_req.info or {}).get("source_doc_id")
        # 只有调用方提供来源文档 ID 时才补充该字段。
        if source_doc_id:
            # 遍历所有抽取出的记忆，统一写入来源文档 ID。
            for memory in flattened_local:
                # 将 source_doc_id 绑定到记忆 metadata，方便后续按文档追踪。
                memory.metadata.source_doc_id = source_doc_id

        # Add memories to text_mem
        # 从全部抽取结果中筛出需要直接写入 text_mem 的普通记忆。
        mem_group = [
            # RawFileMemory 代表原始文件分块，不直接走普通记忆写入列表。
            memory for memory in flattened_local if memory.metadata.memory_type != "RawFileMemory"
        ]

        # Stage 3: write_db
        # 开始数据库写入阶段计时。
        with timed_stage("add", "write_db", cube_id=self.cube_id) as ts_db:
            # 将普通记忆写入底层 text_mem，并拿回生成的 memory IDs。
            mem_ids_local: list[str] = self.naive_mem_cube.text_mem.add(
                # 传入需要写入的普通记忆列表。
                mem_group,
                # 指定写入命名空间，确保记忆进入当前 cube。
                user_name=user_context.mem_cube_id,
            )

            # 记录日志，帮助观察当前分支的执行状态或异常信息。
            self.logger.info(
                f"Added {len(mem_ids_local)} memories for user {add_req.user_id} "
                f"in session {add_req.session_id}: {mem_ids_local}"
            )

            # Add raw file nodes and edges
            # 只有配置保存原始文件且处于 fine 抽取时，才写 RawFileMemory 图节点和边。
            if self.mem_reader.save_rawfile and extract_mode == "fine":
                # 收集所有 RawFileMemory，后续写入图结构。
                raw_file_mem_group = [
                    memory
                    # 遍历当前集合，逐项处理并累积结果。
                    for memory in flattened_local
                    # 只保留原始文件分块类型的记忆。
                    if memory.metadata.memory_type == "RawFileMemory"
                ]
                # 将 RawFileMemory 写成图节点，并和普通记忆 ID 建立关联边。
                self.naive_mem_cube.text_mem.add_rawfile_nodes_n_edges(
                    # 传入原始文件分块记忆。
                    raw_file_mem_group,
                    # 传入普通记忆 ID，用于建立边关系。
                    mem_ids_local,
                    # 传入用户 ID，便于图数据归属。
                    user_id=add_req.user_id,
                    # 指定写入命名空间，确保记忆进入当前 cube。
                    user_name=user_context.mem_cube_id,
                )
            # 把写入的普通记忆数量记录到该阶段指标中。
            ts_db.set(memory_count=len(mem_ids_local))
        # 读取数据库写入阶段耗时。
        write_db_ms = ts_db.duration_ms

        # Stage 4: schedule
        # 开始调度阶段计时。
        with timed_stage("add", "schedule", cube_id=self.cube_id) as ts_sched:
            # 提交后续调度任务，异步/同步模式内部区分。
            self._schedule_memory_tasks(
                # 计算或整理当前步骤所需的中间数据，供后续逻辑使用。
                add_req=add_req,
                # 传入完整用户上下文，避免抽取器重复组装。
                user_context=user_context,
                # 传入本次新写入的记忆 ID 列表。
                mem_ids=mem_ids_local,
                # 记录同步模式。
                sync_mode=sync_mode,
            )

            # Mark merged_from memories as archived when provided in add_req.info
            # 满足特定条件时，对被合并来源记忆做归档处理。
            if (
                # 只有同步模式下才立即处理归档，避免与异步后续流程冲突。
                sync_mode == "sync"
                # 只有 fine 抽取可能产生 merged_from 这样的精细合并信息。
                and extract_mode == "fine"
                and (
                    # 如果没有版本开关属性，按旧逻辑允许归档。
                    not hasattr(self.mem_reader, "memory_version_switch")
                    # 如果版本开关未开启，也按旧逻辑归档 merged_from。
                    or self.mem_reader.memory_version_switch != "on"
                )
            ):
                # 遍历所有抽取出的记忆，统一写入来源文档 ID。
                for memory in flattened_local:
                    # 从每条新记忆 metadata.info 中取出它合并自哪些旧记忆。
                    merged_from = (memory.metadata.info or {}).get("merged_from")
                    # 只有存在来源旧记忆 ID 时才需要归档。
                    if merged_from:
                        # 将 merged_from 统一规范成可遍历的 ID 列表。
                        old_ids = (
                            merged_from
                            # 如果本来就是集合类结构，就直接使用。
                            if isinstance(merged_from, (list | tuple | set))
                            # 如果只是单个 ID，就包装成列表。
                            else [merged_from]
                        )
                        # 只有图数据库可用时才能更新旧记忆节点状态。
                        if self.mem_reader and self.mem_reader.graph_db:
                            # 遍历所有被合并的旧记忆 ID。
                            for old_id in old_ids:
                                # 将可能失败的外部调用包在 try 中，避免单次异常中断整体流程。
                                try:
                                    # 更新图数据库中的旧记忆节点。
                                    self.mem_reader.graph_db.update_node(
                                        # ID 转成字符串，保持图数据库接口入参一致。
                                        str(old_id),
                                        # 将旧节点状态标记为 archived，表示它已被新记忆合并替代。
                                        {"status": "archived"},
                                        # 指定写入命名空间，确保记忆进入当前 cube。
                                        user_name=user_context.mem_cube_id,
                                    )
                                    # 记录日志，帮助观察当前分支的执行状态或异常信息。
                                    self.logger.info(
                                        f"[SingleCubeView] Archived merged_from memory: {old_id}"
                                    )
                                # 单个旧节点归档失败不应中断整个新增流程。
                                except Exception as e:
                                    # 记录日志，帮助观察当前分支的执行状态或异常信息。
                                    self.logger.warning(
                                        f"[SingleCubeView] Failed to archive merged_from memory {old_id}: {e}"
                                    )
                        # 当前分支处理前面条件不成立时的默认路径。
                        else:
                            # 记录日志，帮助观察当前分支的执行状态或异常信息。
                            self.logger.warning(
                                "[SingleCubeView] merged_from provided but graph_db is unavailable; skip archiving."
                            )
        # 读取调度阶段耗时。
        schedule_ms = ts_sched.duration_ms

        # Summary rollup — total_ms is the outer wall-clock, not a new stage
        # 计算从文本记忆处理开始到现在的总耗时，单位毫秒。
        total_ms = int((time.perf_counter() - process_start) * 1000)
        # 统计输入消息数量，空消息时记为 0。
        input_msg_count = len(add_req.messages) if add_req.messages else 0
        # 统计成功写入的普通记忆数量。
        memory_count = len(mem_ids_local)
        # 粗略估算输入 token 数，用于性能和成本分析。
        est_input_tokens = (
            # 将每条消息内容长度累加。
            sum(
                # 字典消息取 content 长度，非字典消息转字符串后计算长度。
                len(str(m.get("content", ""))) if isinstance(m, dict) else len(str(m))
                # 遍历所有输入消息；为空时遍历空列表。
                for m in (add_req.messages or [])
            )
            # 用字符数除以 4 粗略估算 token 数。
            // 4
        )
        # 立即输出 summary 阶段指标，把各阶段耗时和数量汇总到日志/监控。
        timed_stage.emit_now(
            # 指标所属业务域为 add。
            "add",
            # 指标阶段名为 summary。
            "summary",
            # 标记指标来自当前 cube。
            cube_id=self.cube_id,
            # 记录同步模式。
            sync_mode=sync_mode,
            # 记录抽取模式。
            extract_mode=extract_mode,
            # 记录输入消息数量。
            input_msg_count=input_msg_count,
            # 记录估算 token 数。
            est_input_tokens=est_input_tokens,
            # 记录写入记忆数量。
            memory_count=memory_count,
            # 记录抽取阶段耗时。
            get_memory_ms=get_memory_ms,
            # 记录写库阶段耗时。
            write_db_ms=write_db_ms,
            # 记录调度阶段耗时。
            schedule_ms=schedule_ms,
            # 记录整体耗时。
            total_ms=total_ms,
            # 计算平均每条记忆耗时，max(..., 1) 避免除零。
            per_item_ms=total_ms // max(memory_count, 1),
        )

        # Format results uniformly
        # 构造 API 返回所需的简洁记忆结果列表。
        text_memories = [
            {
                # 返回记忆文本。
                "memory": memory.memory,
                # 返回底层写入后生成的记忆 ID。
                "memory_id": memory_id,
                # 返回记忆类型，便于调用方区分普通记忆等类型。
                "memory_type": memory.metadata.memory_type,
                # 返回当前 cube_id，便于多 cube 场景追踪来源。
                "cube_id": self.cube_id,
            }
            # 将写入 ID 与原记忆对象配对；strict=False 允许长度不完全一致时不抛错。
            for memory_id, memory in zip(mem_ids_local, mem_group, strict=False)
        ]

        # 返回本次新增的格式化记忆结果。
        return text_memories
