"""
Redis Queue implementation for SchedulerMessageItem objects.

This module provides a Redis-based queue implementation that can replace
the local memos_message_queue functionality in BaseScheduler.
"""

# 从上下文模块中读取当前请求的 API 路径和 trace_id。
# 队列入队时会把这些上下文写回 ScheduleMessageItem，方便后续消费者日志与原始请求关联。
from memos.context.context import get_current_api_path, get_current_trace_id

# 获取项目统一 logger。
# 这里用于记录队列初始化、入队、跳过 handler、ack 等关键流程。
from memos.log import get_logger

# 调度系统中的消息模型。
# ScheduleTaskQueue 接收和输出的核心对象都是 ScheduleMessageItem。
from memos.mem_scheduler.schemas.message_schemas import ScheduleMessageItem

# 本地队列实现。
# 当 use_redis_queue=False 时，ScheduleTaskQueue 会用它作为底层队列。
from memos.mem_scheduler.task_schedule_modules.local_queue import SchedulerLocalQueue

# 调度编排器。
# Redis 队列会使用 orchestrator 管理流、消费者组或相关调度协作逻辑。
from memos.mem_scheduler.task_schedule_modules.orchestrator import SchedulerOrchestrator

# Redis 队列实现。
# 当 use_redis_queue=True 时，ScheduleTaskQueue 会把消息写入 Redis Stream。
from memos.mem_scheduler.task_schedule_modules.redis_queue import SchedulerRedisQueue

# 获取当前 UTC 时间。
# 入队前如果消息没有 timestamp，会用它补齐时间戳。
from memos.mem_scheduler.utils.db_utils import get_utc_now

# 按 user_id 和 mem_cube_id 对消息进行分组。
# 批量提交消息时会用它保持同一用户、同一 cube 的消息处理结构更清晰。
from memos.mem_scheduler.utils.misc_utils import group_messages_by_user_and_mem_cube

# emit_monitor_event 用于上报调度监控事件。
# to_iso 用于把 timestamp 转成统一的 ISO 字符串格式。
from memos.mem_scheduler.utils.monitor_event_utils import emit_monitor_event, to_iso

# 任务状态追踪器。
# Redis 队列可以通过它记录任务状态变化，例如排队、处理中、完成或失败。
from memos.mem_scheduler.utils.status_tracker import TaskStatusTracker


# 当前模块的 logger。
# 使用 __name__ 能在日志中体现具体模块来源。
logger = get_logger(__name__)


# ScheduleTaskQueue 是调度队列的统一包装层。
# 它屏蔽底层队列实现差异，让上层 scheduler 可以用同一套 submit/get/clear/qsize 接口。
class ScheduleTaskQueue:
    # 初始化队列包装器。
    # 根据 use_redis_queue 决定使用 Redis 队列还是本地队列。
    def __init__(
        self,
        use_redis_queue: bool,
        maxsize: int,
        disabled_handlers: list | None = None,
        orchestrator: SchedulerOrchestrator | None = None,
        status_tracker: TaskStatusTracker | None = None,
    ):
        # 保存是否启用 Redis 队列的开关。
        # 后续日志和行为分支都会依赖这个标记。
        self.use_redis_queue = use_redis_queue

        # 保存队列最大长度配置。
        # 本地队列直接使用它；Redis 队列会进一步校验后转换成 max_len。
        self.maxsize = maxsize

        # 如果外部没有传 orchestrator，就创建一个默认的 SchedulerOrchestrator。
        # 这样 Redis 队列始终有编排器可用，同时也允许测试时注入 mock/stub。
        self.orchestrator = SchedulerOrchestrator() if orchestrator is None else orchestrator

        # 保存任务状态追踪器。
        # 它可以在初始化 Redis 队列时向下传递，也可以后续通过 set_status_tracker 补充注入。
        self.status_tracker = status_tracker

        # 如果启用 Redis 队列，则创建 SchedulerRedisQueue。
        # 这种模式适合多进程、多实例或需要持久化/跨服务消费的调度场景。
        if self.use_redis_queue:
            # Redis Stream 的 max_len 应该是正整数。
            # 如果调用方传入 None、非 int 或小于等于 0，则视为不限制长度。
            if maxsize is None or not isinstance(maxsize, int) or maxsize <= 0:
                maxsize = None

            # 初始化 Redis 队列实现。
            # consumer_group 和 consumer_name 定义 Redis Stream 消费者组语义。
            self.memos_message_queue = SchedulerRedisQueue(
                # Redis Stream 最大长度；None 表示不主动限制。
                max_len=maxsize,

                # Redis 消费者组名称，用于多个消费者协作读取同一组 stream。
                consumer_group="scheduler_group",

                # 当前消费者名称。
                # 如果后续有多个消费者实例，通常需要保证名称区分实例。
                consumer_name="scheduler_consumer",

                # 将编排器传入 Redis 队列，供队列内部做 stream 管理或调度协作。
                orchestrator=self.orchestrator,

                # Propagate status_tracker
                # 把状态追踪器传给 Redis 队列，让底层队列也能更新任务状态。
                status_tracker=self.status_tracker,
            )

        # 如果不启用 Redis，则创建本地内存队列。
        # 这种模式适合单进程、本地开发或不需要跨进程消费的场景。
        else:
            self.memos_message_queue = SchedulerLocalQueue(maxsize=self.maxsize)

        # 保存禁用的 handler label 列表。
        # submit_messages 时如果消息 label 在这个列表里，会跳过入队。
        self.disabled_handlers = disabled_handlers

        # 记录队列包装器初始化结果。
        # stream_prefix 只有 Redis 队列通常会有；本地队列没有时返回 None。
        logger.info(
            "[SCHEDULE_TASK_QUEUE] Initialized queue wrapper. use_redis_queue=%s, queue_type=%s, stream_prefix=%s",
            self.use_redis_queue,
            type(self.memos_message_queue).__name__,
            getattr(self.memos_message_queue, "stream_key_prefix", None),
        )

    # 后置设置状态追踪器。
    # 用于初始化时 tracker 尚不可用，但队列创建后又需要把 tracker 注入到底层队列的情况。
    def set_status_tracker(self, status_tracker: TaskStatusTracker) -> None:
        """
        Set the status tracker for this queue and propagate it to the underlying queue implementation.

        This allows the tracker to be injected after initialization (e.g., when Redis connection becomes available).
        """
        # 更新包装器自身保存的 tracker。
        self.status_tracker = status_tracker

        # 如果底层队列存在，并且它暴露了 status_tracker 属性，就同步更新。
        # 这样上层不需要区分 Redis 队列或本地队列是否在初始化时拿到了 tracker。
        if self.memos_message_queue and hasattr(self.memos_message_queue, "status_tracker"):
            # SchedulerRedisQueue has status_tracker attribute (from our previous fix)
            # SchedulerLocalQueue can also accept it dynamically if it doesn't use __slots__
            # 动态给底层队列设置 status_tracker。
            self.memos_message_queue.status_tracker = status_tracker

            # 记录 tracker 已经传播到底层队列。
            logger.info("Propagated status_tracker to underlying message queue")

    # 确认 Redis 消息已被处理。
    # ack 只对 Redis 队列有意义，本地队列没有 Redis Stream 的待确认消息概念。
    def ack_message(
        self,
        user_id: str,
        mem_cube_id: str,
        task_label: str,
        redis_message_id,
        message: ScheduleMessageItem | None,
    ) -> None:
        # 如果当前底层队列不是 Redis 队列，则无法 ack。
        # 这里选择记录 warning 并返回，而不是抛异常，避免调用方在本地模式下崩溃。
        if not isinstance(self.memos_message_queue, SchedulerRedisQueue):
            logger.warning("ack_message is only supported for Redis queues")
            return

        # 将 ack 请求转发给 Redis 队列实现。
        # user_id、mem_cube_id、task_label 用于定位具体 stream，redis_message_id 用于确认具体消息。
        self.memos_message_queue.ack_message(
            user_id=user_id,
            mem_cube_id=mem_cube_id,
            task_label=task_label,
            redis_message_id=redis_message_id,
            message=message,
        )

    # 获取当前底层队列中涉及的 stream key。
    # Redis 模式直接问 Redis 队列；本地模式从本地 queue_streams 字典中取 key。
    def get_stream_keys(self) -> list[str]:
        # Redis 队列有专门的 get_stream_keys 方法，可能会从 Redis 或内部索引读取。
        if isinstance(self.memos_message_queue, SchedulerRedisQueue):
            stream_keys = self.memos_message_queue.get_stream_keys()

        # 本地队列没有 Redis Stream，但内部也按 stream_key 组织 queue_streams。
        # 因此直接返回本地队列字典的 keys。
        else:
            stream_keys = list(self.memos_message_queue.queue_streams.keys())

        # 返回统一的 stream key 列表。
        return stream_keys

    # 提交一条或多条调度消息到底层队列。
    # 这是 ScheduleTaskQueue 最核心的方法：补上下文、生成 stream_key、过滤禁用 handler、上报监控事件、执行 put。
    def submit_messages(self, messages: ScheduleMessageItem | list[ScheduleMessageItem]):
        """Submit messages to the message queue (either local queue or Redis)."""
        # 允许调用方传单个 ScheduleMessageItem。
        # 为了后续统一处理，单条消息会被包装成列表。
        if isinstance(messages, ScheduleMessageItem):
            messages = [messages]

        # 空消息列表是异常使用方式。
        # 这里记录 error 并返回，不继续执行后续逻辑。
        if len(messages) < 1:
            logger.error("submit_messages called with empty payload")
            return

        # 从当前上下文读取 trace_id。
        # 该 trace_id 通常来自一次 API 请求，用于把入队、消费、执行日志串起来。
        current_trace_id = get_current_trace_id()

        # 从当前上下文读取 API path。
        # 后续消息处理或监控中可以知道该任务是由哪个接口触发。
        current_api_path = get_current_api_path()

        # 遍历每条消息，在真正入队前补充公共上下文字段。
        for msg in messages:
            # 如果当前请求上下文里有 trace_id，则优先覆盖消息自身的 trace_id。
            # 这样队列任务能与当前请求链路保持一致。
            if current_trace_id:
                # Prefer current request trace_id so logs can be correlated
                msg.trace_id = current_trace_id

            # 如果当前上下文有 API path，并且消息自身尚未设置 api_path，就写入它。
            # 这里不覆盖已有 api_path，保留上游显式设置的来源信息。
            if current_api_path and not getattr(msg, "api_path", None):
                msg.api_path = current_api_path

            # 根据 user_id、mem_cube_id 和 task label 生成 stream_key。
            # 这个 key 决定消息会进入哪个 Redis Stream 或本地分流队列。
            msg.stream_key = self.memos_message_queue.get_stream_key(
                user_id=msg.user_id, mem_cube_id=msg.mem_cube_id, task_label=msg.label
            )

        # 单条消息走简化路径。
        # 避免不必要的分组，同时逻辑更直接。
        if len(messages) == 1:
            # 如果消息还没有 timestamp，则补当前 UTC 时间。
            # 该时间用于监控 enqueue_ts，也可能用于调度排序或排查延迟。
            if getattr(messages[0], "timestamp", None) is None:
                messages[0].timestamp = get_utc_now()

            # 如果该消息 label 在禁用 handler 列表中，则跳过入队。
            # disabled_handlers 常用于临时关闭某类任务处理。
            if self.disabled_handlers and messages[0].label in self.disabled_handlers:
                # debug 日志记录被跳过的任务 label、item、用户和 cube。
                logger.debug(
                    "Skip disabled handler. label=%s item_id=%s user_id=%s mem_cube_id=%s",
                    messages[0].label,
                    messages[0].item_id,
                    messages[0].user_id,
                    messages[0].mem_cube_id,
                )

            # 如果没有被禁用，则上报 enqueue 事件并真正入队。
            else:
                # 把 timestamp 转成 ISO 字符串，统一监控事件字段格式。
                enqueue_ts = to_iso(getattr(messages[0], "timestamp", None))

                # 上报 enqueue 监控事件。
                # event_duration_ms 和 total_duration_ms 在入队时初始化为 0。
                emit_monitor_event(
                    "enqueue",
                    messages[0],
                    {"enqueue_ts": enqueue_ts, "event_duration_ms": 0, "total_duration_ms": 0},
                )

                # 将消息放入底层队列。
                # 这里会根据底层实现写入 Redis Stream 或本地队列。
                self.memos_message_queue.put(messages[0])

        # 多条消息走批量路径。
        # 该路径会先按用户和 memory cube 分组，再逐条入队。
        else:
            # 按 user_id 和 mem_cube_id 对消息分组。
            # 这样可以让后续循环体现任务归属，也方便未来按用户/cube 做限流或排序。
            user_cube_groups = group_messages_by_user_and_mem_cube(messages)

            # Process each user and mem_cube combination
            # 第一层遍历 user_id。
            for _user_id, cube_groups in user_cube_groups.items():
                # 第二层遍历该用户下的 mem_cube_id。
                for _mem_cube_id, user_cube_msgs in cube_groups.items():
                    # 第三层遍历同一 user + cube 下的具体消息。
                    for message in user_cube_msgs:
                        # 批量路径中再次做类型校验。
                        # 这可以防止调用方传入混杂列表，导致底层队列写入异常对象。
                        if not isinstance(message, ScheduleMessageItem):
                            error_msg = f"Invalid message type: {type(message)}, expected ScheduleMessageItem"

                            # 记录错误，再抛出 TypeError，让调用方知道这是明确的参数错误。
                            logger.error(error_msg)
                            raise TypeError(error_msg)

                        # 如果消息没有 timestamp，则补当前 UTC 时间。
                        if getattr(message, "timestamp", None) is None:
                            message.timestamp = get_utc_now()

                        # 如果该任务 label 被禁用，则跳过这条消息。
                        # continue 表示同一批次中的其他消息仍会继续处理。
                        if self.disabled_handlers and message.label in self.disabled_handlers:
                            logger.debug(
                                "Skip disabled handler. label=%s item_id=%s user_id=%s mem_cube_id=%s",
                                message.label,
                                message.item_id,
                                message.user_id,
                                message.mem_cube_id,
                            )
                            continue

                        # 将入队时间转成 ISO 格式，供监控事件使用。
                        enqueue_ts = to_iso(getattr(message, "timestamp", None))

                        # 上报当前消息的 enqueue 事件。
                        # 批量提交中每条消息仍然独立上报，方便按 item 或 task 追踪。
                        emit_monitor_event(
                            "enqueue",
                            message,
                            {
                                "enqueue_ts": enqueue_ts,
                                "event_duration_ms": 0,
                                "total_duration_ms": 0,
                            },
                        )

                        # 将当前消息放入底层队列。
                        self.memos_message_queue.put(message)

        # 入队流程结束后记录汇总日志。
        # 注意 total 是原始 messages 的数量，包括可能被 disabled_handlers 跳过的消息。
        logger.info(
            "Queue submit completed. backend=%s total=%s",
            "redis_queue" if self.use_redis_queue else "local_queue",
            len(messages),
        )

    # 从底层队列拉取一批待处理消息。
    # batch_size 由 scheduler 消费端决定，用于控制每轮处理量。
    def get_messages(self, batch_size: int) -> list[ScheduleMessageItem]:
        # 直接转发到底层队列实现。
        # Redis 队列可能会从 stream/consumer group 拉取；本地队列则从内存结构取出。
        return self.memos_message_queue.get_messages(batch_size=batch_size)

    # 清空底层队列。
    # 常用于测试、重置或停止前清理。
    def clear(self):
        # 由底层队列决定具体清理方式。
        self.memos_message_queue.clear()

    # 返回当前队列大小。
    # 对 Redis 队列和本地队列，qsize 的具体统计方式由各自实现负责。
    def qsize(self):
        # 统一暴露队列大小接口给上层 scheduler。
        return self.memos_message_queue.qsize()
