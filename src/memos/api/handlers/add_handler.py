"""
Add handler for memory addition functionality (Class-based version).

This module provides a class-based implementation of add handlers,
using dependency injection for better modularity and testability.
"""

# 从 Pydantic 引入 validate_call，用于在运行时校验函数入参是否符合类型声明。
# 这里后面会用它校验 messages 是否满足 MessageList 的结构要求。
from pydantic import validate_call

# 引入所有 Handler 的基础类和依赖容器。
# BaseHandler 提供公共能力，例如日志、依赖校验、依赖属性访问等。
# HandlerDependencies 用于把外部依赖集中注入进 Handler，方便测试和替换实现。
from memos.api.handlers.base_handler import BaseHandler, HandlerDependencies

# APIADDRequest 是新增记忆接口的请求模型。
# APIFeedbackRequest 是反馈型记忆写入接口的请求模型。
# MemoryResponse 是统一返回给 API 层的响应模型。
from memos.api.product_models import APIADDRequest, APIFeedbackRequest, MemoryResponse

# list_all_fields 用于拿到文本记忆 item 中已有的字段名。
# 后续会用这些字段名过滤 add_req.info，避免用户传入的 info 覆盖系统保留字段。
from memos.memories.textual.item import (
    list_all_fields,
)

# CompositeCubeView 表示多个 memory cube 的组合视图。
# 当一次写入需要落到多个 cube 时，会通过它统一分发请求。
from memos.multi_mem_cube.composite_cube import CompositeCubeView

# SingleCubeView 表示单个 memory cube 的操作视图。
# 当目标 cube 只有一个时，直接使用它执行新增或反馈处理。
from memos.multi_mem_cube.single_cube import SingleCubeView

# MemCubeView 是 cube view 的抽象类型。
# _build_cube_view 会根据目标 cube 数量返回 SingleCubeView 或 CompositeCubeView，
# 因此这里用该抽象类型标注返回值。
from memos.multi_mem_cube.views import MemCubeView

# hookable 装饰器用于把 handler 方法挂到插件钩子系统中。
# 这里的 "add" 表示该方法对应 add 这个生命周期或业务钩子。
from memos.plugins.hooks import hookable

# MessageList 是消息列表的类型定义。
# 它通常约束每条消息应包含 role、content 等对话字段。
from memos.types import MessageList


# AddHandler 继承自 BaseHandler，专门处理“新增记忆”相关的 API 请求。
class AddHandler(BaseHandler):
    """
    Handler for memory addition operations.

    Handles text memory additions with sync/async support.
    """

    # 构造函数接收一组外部依赖，而不是在类内部自行创建依赖。
    # 这样可以让 Handler 更容易测试，也能在不同运行环境中替换依赖实现。
    def __init__(self, dependencies: HandlerDependencies):
        """
        Initialize add handler.

        Args:
            dependencies: HandlerDependencies instance
        """
        # 先调用父类构造函数，把依赖挂载到 BaseHandler 提供的属性上。
        # 例如 self.naive_mem_cube、self.mem_reader、self.logger 等通常在这里初始化。
        super().__init__(dependencies)

        # 校验当前 Handler 运行所必须的依赖是否已经注入。
        # 如果缺少任何一项，应尽早在初始化阶段失败，而不是等到处理请求时才出错。
        self._validate_dependencies(
            "naive_mem_cube", "mem_reader", "mem_scheduler", "feedback_server"
        )

    # 将 handle_add_memories 注册为 add 钩子的可 hook 方法。
    # 外部插件可以围绕该方法做前置、后置或替换逻辑。
    @hookable("add")
    # 主入口方法：处理新增记忆接口请求。
    # 它会根据请求内容决定走普通新增流程，还是走 feedback 记忆处理流程。
    def handle_add_memories(self, add_req: APIADDRequest) -> MemoryResponse:
        """
        Main handler for add memories endpoint.

        Orchestrates the addition of text memories,
        supporting concurrent processing.

        Args:
            add_req: Add memory request (deprecated fields are converted in model validator)

        Returns:
            MemoryResponse with added memory information
        """
        # 打一条诊断日志，记录请求已经进入 AddHandler。
        # model_dump_json(indent=2) 会把 Pydantic 模型转成格式化 JSON，便于排查问题。
        # 这里日志里带有修改时间，通常用于定位线上部署版本或临时诊断版本。
        self.logger.info(
            f"[DIAGNOSTIC] server_router -> add_handler.handle_add_memories called (Modified at 2025-11-29 18:46). Full request: {add_req.model_dump_json(indent=2)}"
        )

        # 如果请求中携带了 info 字段，需要先做保护性过滤。
        # info 一般用于存放用户自定义元信息，但不能覆盖系统记忆 item 的保留字段。
        if add_req.info:
            # 获取文本记忆 item 支持或保留的全部字段名。
            # 这些字段不允许出现在 info 中，避免后续写入时产生字段冲突。
            exclude_fields = list_all_fields()

            # 记录过滤前 info 的字段数量。
            # 后面通过数量变化判断是否真的过滤掉了非法字段。
            info_len = len(add_req.info)

            # 重建 info 字典，只保留不在系统保留字段列表中的键值对。
            # 这一步不会改变合法的用户自定义字段，只移除可能污染核心模型字段的内容。
            add_req.info = {k: v for k, v in add_req.info.items() if k not in exclude_fields}

            # 如果过滤后字段数量减少，说明用户传入了不允许出现在 info 中的字段。
            if len(add_req.info) < info_len:
                # 记录 warning，提醒调用方或开发者 info 中包含了非法字段。
                # 这里不直接抛错，而是选择静默移除并告警，使接口更宽容。
                self.logger.warning(f"[AddHandler] info fields can not contain {exclude_fields}.")

        # 根据请求中的 writable_cube_ids 或 user_id 构建 cube view。
        # 后续无论是普通新增还是 feedback 处理，都通过这个抽象 view 操作底层 cube。
        cube_view = self._build_cube_view(add_req)

        # 定义一个局部校验函数，用于触发 Pydantic 对 messages 的类型校验。
        # 之所以函数体为空，是因为校验目的只在入参进入函数之前完成。
        @validate_call
        def _check_messages(messages: MessageList) -> None:
            # validate_call 会在真正执行函数体之前校验 messages。
            # 如果结构不符合 MessageList，会直接抛出校验异常。
            pass

        # 如果当前请求被标记为 feedback，则优先尝试走反馈记忆流程。
        # feedback 与普通 add 的区别在于：它会从对话中抽取最后一条用户反馈作为反馈内容。
        if add_req.is_feedback:
            try:
                # 取出本次请求携带的消息列表。
                # 这些消息通常表示当前轮或最近几轮对话内容。
                messages = add_req.messages

                # 对 messages 做结构校验，确保后面可以安全访问 role/content 字段。
                _check_messages(messages)

                # 如果请求中有历史对话 chat_history，则使用它；
                # 如果没有，则退化为空列表，保证后续拼接逻辑稳定。
                chat_history = add_req.chat_history if add_req.chat_history else []

                # 将历史对话和当前消息拼接成完整上下文。
                # 后续会在这个完整上下文中寻找最后一条 user 消息作为反馈内容。
                concatenate_chat = chat_history + messages

                # 从完整上下文中找到最后一条 role 为 user 的消息下标。
                # max(...) 保证使用“最后一次用户表达”作为本次反馈内容。
                # 如果上下文中没有 user 消息，这里会抛出 ValueError 并进入 except。
                last_user_index = max(
                    i for i, d in enumerate(concatenate_chat) if d["role"] == "user"
                )

                # 最后一条用户消息的 content 被视为本次 feedback 的核心内容。
                # 换句话说，用户最后说的话就是要反馈给记忆系统的信息。
                feedback_content = concatenate_chat[last_user_index]["content"]

                # 最后一条用户消息之前的所有内容被视为反馈历史。
                # 这些历史用于帮助 feedback_server 理解反馈发生的上下文。
                feedback_history = concatenate_chat[:last_user_index]

                # 将 add 请求转换成 feedback 请求。
                # 这里保留用户、会话、任务、目标 cube、异步模式和 info 等通用字段。
                feedback_req = APIFeedbackRequest(
                    # 用户 ID 用于区分不同用户的记忆空间。
                    user_id=add_req.user_id,

                    # 会话 ID 用于标识反馈来自哪一次对话或交互上下文。
                    session_id=add_req.session_id,

                    # 任务 ID 用于关联上游任务，便于追踪和调度。
                    task_id=add_req.task_id,

                    # history 不包含最后一条用户反馈本身，只包含它之前的上下文。
                    history=feedback_history,

                    # feedback_content 是实际要交给反馈记忆流程处理的文本。
                    feedback_content=feedback_content,

                    # writable_cube_ids 指定本次反馈可以写入哪些 cube。
                    writable_cube_ids=add_req.writable_cube_ids,

                    # async_mode 决定底层处理是否采用异步模式。
                    async_mode=add_req.async_mode,

                    # info 携带额外元信息；上面已经过滤过保留字段。
                    info=add_req.info,
                )

                # 通过 cube_view 执行反馈记忆写入。
                # 如果 cube_view 是 CompositeCubeView，则内部会分发到多个 cube。
                # 如果是 SingleCubeView，则只写入单个 cube。
                process_record = cube_view.feedback_memories(feedback_req)

                # 记录反馈处理得到的结果数量。
                # 注意这里假设 process_record 支持 len()，否则会触发异常并进入 except。
                self.logger.info(
                    f"[ADDFeedbackHandler] Final feedback results count={len(process_record)}"
                )

                # feedback 流程成功后直接返回，不再继续走普通 add 流程。
                return MemoryResponse(
                    # 返回给调用方的提示消息，说明反馈型记忆处理成功。
                    message="Memory feedback successfully",

                    # data 包装成列表，保持与普通新增返回结构接近。
                    # 这里的 process_record 通常是一组反馈处理结果或处理记录。
                    data=[process_record],
                )

            # 捕获 feedback 流程中的所有异常。
            # 当前实现不会把异常继续抛出，而是记录 warning 后回退到普通 add 流程。
            except Exception as e:
                # 记录 feedback 失败原因，便于排查。
                # 因为没有 return 或 raise，后续会继续执行普通 add_memories。
                self.logger.warning(f"[ADDFeedbackHandler] Running error: {e}")

        # 如果不是 feedback 请求，或者 feedback 流程失败，
        # 则走普通新增记忆流程。
        results = cube_view.add_memories(add_req)

        # 记录普通新增记忆的最终结果数量。
        # results 通常是写入成功的 memory 信息列表或处理记录列表。
        self.logger.info(f"[AddHandler] Final add results count={len(results)}")

        # 构造统一响应，返回普通新增记忆的处理结果。
        return MemoryResponse(
            # 返回给调用方的提示消息，说明普通记忆新增成功。
            message="Memory added successfully",

            # data 中放入底层 cube_view.add_memories 的返回结果。
            data=results,
        )

    # 解析本次请求应该写入哪些 cube。
    # 该方法把“目标 cube 的选择逻辑”从主流程中拆出来，便于复用和测试。
    def _resolve_cube_ids(self, add_req: APIADDRequest) -> list[str]:
        """
        Normalize target cube ids from add_req.
        Priority:
        1) writable_cube_ids (deprecated mem_cube_id is converted to this in model validator)
        2) fallback to user_id
        """
        # 如果请求显式指定了 writable_cube_ids，则优先使用它。
        # 这表示调用方希望本次记忆写入一个或多个指定 cube。
        if add_req.writable_cube_ids:
            # 使用 dict.fromkeys 去重，同时保留原始顺序。
            # 再转回 list，得到稳定且不重复的 cube_id 列表。
            return list(dict.fromkeys(add_req.writable_cube_ids))

        # 如果没有显式指定 writable_cube_ids，则退回到 user_id。
        # 这相当于默认把用户 ID 当作该用户自己的 memory cube ID。
        return [add_req.user_id]

    # 根据解析出的 cube_id 数量，构造合适的 MemCubeView。
    # 单 cube 使用 SingleCubeView，多 cube 使用 CompositeCubeView。
    def _build_cube_view(self, add_req: APIADDRequest) -> MemCubeView:
        # 先统一解析目标 cube IDs，避免主流程关心字段优先级和去重细节。
        cube_ids = self._resolve_cube_ids(add_req)

        # 如果只有一个目标 cube，则无需组合视图，直接构建单 cube 视图。
        if len(cube_ids) == 1:
            # 取出唯一的 cube_id，后续所有记忆操作都针对这个 cube。
            cube_id = cube_ids[0]

            # 创建 SingleCubeView。
            # 它封装了单个 cube 的新增、反馈、调度等操作入口。
            return SingleCubeView(
                # 当前要操作的 cube 标识。
                cube_id=cube_id,

                # 底层 naive memory cube 实例，负责实际存储或管理记忆。
                naive_mem_cube=self.naive_mem_cube,

                # mem_reader 用于读取或解析输入中的可记忆内容。
                mem_reader=self.mem_reader,

                # mem_scheduler 用于调度同步或异步记忆处理任务。
                mem_scheduler=self.mem_scheduler,

                # 将 handler 的 logger 传入 view，保持日志上下文一致。
                logger=self.logger,

                # feedback_server 用于处理反馈型记忆逻辑。
                feedback_server=self.feedback_server,

                # searcher 当前显式传 None，说明此视图暂不注入搜索组件。
                searcher=None,
            )

        # 如果目标 cube 超过一个，则为每个 cube 构建单独的 SingleCubeView，
        # 再用 CompositeCubeView 统一包装。
        else:
            # 使用列表推导式为每个 cube_id 创建一个 SingleCubeView。
            # 每个 SingleCubeView 共享同一组依赖，但绑定不同的 cube_id。
            single_views = [
                SingleCubeView(
                    # 当前循环中的 cube 标识。
                    cube_id=cube_id,

                    # 共享底层 memory cube 依赖。
                    naive_mem_cube=self.naive_mem_cube,

                    # 共享记忆读取器。
                    mem_reader=self.mem_reader,

                    # 共享调度器。
                    mem_scheduler=self.mem_scheduler,

                    # 共享日志对象。
                    logger=self.logger,

                    # 共享反馈服务。
                    feedback_server=self.feedback_server,

                    # 当前同样不注入搜索组件。
                    searcher=None,
                )
                # 遍历所有目标 cube_id，为每个目标构造一个单 cube view。
                for cube_id in cube_ids
            ]

            # 使用 CompositeCubeView 包装多个 SingleCubeView。
            # 这样上层只需要调用 add_memories 或 feedback_memories 一次，
            # 具体分发到多个 cube 的细节由组合视图负责。
            return CompositeCubeView(
                # 多个单 cube view 共同组成一个复合视图。
                cube_views=single_views,

                # 传入同一个 logger，便于组合视图记录整体处理日志。
                logger=self.logger,
            )
