# 引入 concurrent 包，后面会使用 concurrent.futures.wait 等方法等待线程池任务完成。
import concurrent

# threading 用于创建锁、事件类型以及管理与线程相关的并发状态。
# 当前类中主要通过 threading.Lock 保护 _running_tasks 和 _futures 等共享数据。
import threading

# time 用于记录任务开始、结束、排队等待等时间点。
# 这些时间会被用于 metrics 和 monitor event 的耗时统计。
import time

# defaultdict 用于按 label 对消息进行分组时自动创建 list。
from collections import defaultdict

# Callable 用于标注 handler、回调函数和外部传入的任务函数类型。
from collections.abc import Callable

# datetime 和 timezone 用于生成 UTC ISO 时间戳。
# 调度监控事件中的 start_ts、finish_ts、dequeue_ts 都依赖它们。
from datetime import datetime, timezone

# Any 用于表示无法提前确定的返回值或外部依赖对象类型。
from typing import Any

# 导入上下文感知的线程池、请求上下文和 trace 工具。
# dispatcher 会在 handler 执行前恢复 trace_id/api_path/user_name，保证异步线程中的日志仍能关联原请求。
from memos.context.context import (
    ContextThreadPoolExecutor,
    RequestContext,
    generate_trace_id,
    set_request_context,
)

# 获取项目统一 logger。
from memos.log import get_logger

# BaseSchedulerModule 是调度模块的公共基类。
# SchedulerDispatcher 继承它，从而接入调度系统的基础模块能力。
from memos.mem_scheduler.general_modules.base import BaseSchedulerModule

# ThreadManager 提供更高层的多线程任务执行能力。
# 本类用它支持“竞争式任务执行”和“批量并发任务执行”。
from memos.mem_scheduler.general_modules.task_threads import ThreadManager

# DEFAULT_STOP_WAIT 是关闭线程池时的默认等待配置。
from memos.mem_scheduler.schemas.general_schemas import (
    DEFAULT_STOP_WAIT,
)

# ScheduleMessageItem 是调度系统中被 dispatcher 分发的消息对象。
# ScheduleLogForWebItem 是向 Web 端提交任务状态日志时使用的数据结构。
from memos.mem_scheduler.schemas.message_schemas import ScheduleLogForWebItem, ScheduleMessageItem

# RunningTaskItem 用于记录一个正在执行的调度任务。
# TaskPriorityLevel 用于注册 handler 时配置任务优先级。
from memos.mem_scheduler.schemas.task_schemas import RunningTaskItem, TaskPriorityLevel

# SchedulerOrchestrator 用于记录每类 task label 的调度配置，例如优先级、最小空闲时间等。
from memos.mem_scheduler.task_schedule_modules.orchestrator import SchedulerOrchestrator

# SchedulerRedisQueue 是 Redis 队列实现。
# dispatcher 在 handler 执行结束的 finally 中会识别 Redis 队列并 ack 消息。
from memos.mem_scheduler.task_schedule_modules.redis_queue import SchedulerRedisQueue

# ScheduleTaskQueue 是队列包装器。
# __init__ 支持传入这个包装器，也支持传入具体队列实例。
from memos.mem_scheduler.task_schedule_modules.task_queue import ScheduleTaskQueue

# group_messages_by_user_and_mem_cube 用于将消息先按 user_id/mem_cube_id 分组。
# is_playground_api 用于避免 playground 请求产生 Web 任务状态日志。
from memos.mem_scheduler.utils.misc_utils import (
    group_messages_by_user_and_mem_cube,
    is_playground_api,
)

# emit_monitor_event 用于上报 enqueue/dequeue/start/finish 等调度监控事件。
# to_iso 用于把不同类型的时间戳统一转成 ISO 字符串。
from memos.mem_scheduler.utils.monitor_event_utils import emit_monitor_event, to_iso

# TaskStatusTracker 负责记录任务提交、开始、完成、失败等状态。
# dispatcher 会在 handler 包装器里调用它更新状态。
from memos.mem_scheduler.utils.status_tracker import TaskStatusTracker


# 创建当前模块 logger。
logger = get_logger(__name__)


# SchedulerDispatcher 是消息调度分发器。
# 它接收队列消费者拉取到的 ScheduleMessageItem，按 user/cube/label 分组，并派发给注册的 handler。
class SchedulerDispatcher(BaseSchedulerModule):
    """
    Thread pool-based message dispatcher that routes messages to dedicated handlers
    based on their labels.

    Features:
    - Dedicated thread pool per message label
    - Batch message processing
    - Graceful shutdown
    - Bulk handler registration
    - Thread race competition for parallel task execution
    """

    # 初始化 dispatcher。
    # 这里完成线程池、handler 注册表、任务追踪表、状态追踪器、metrics、Web 日志回调等核心组件的准备。
    def __init__(
        self,
        max_workers: int = 30,
        memos_message_queue: ScheduleTaskQueue | None = None,
        enable_parallel_dispatch: bool = True,
        config=None,
        status_tracker: TaskStatusTracker | None = None,
        metrics: Any | None = None,
        submit_web_logs: Callable | None = None,  # ADDED
        orchestrator: SchedulerOrchestrator | None = None,
    ):
        # 调用调度模块基类初始化逻辑。
        super().__init__()

        # 保存配置对象或配置字典。
        # 后续会从中读取 multi_task_running_timeout、stop_wait 等参数。
        self.config = config

        # Main dispatcher thread pool
        # 保存 dispatcher 主线程池最大 worker 数。
        self.max_workers = max_workers

        # Accept either a ScheduleTaskQueue wrapper or a concrete queue instance
        # 如果传入的是 ScheduleTaskQueue 包装器，则取其内部真实队列对象。
        # 如果传入的已经是具体队列实例，则直接保存。
        self.memos_message_queue = (
            memos_message_queue.memos_message_queue
            if hasattr(memos_message_queue, "memos_message_queue")
            else memos_message_queue
        )

        # 保存 orchestrator。
        # 没有外部注入时创建默认实例，用于维护 task label 的优先级配置。
        self.orchestrator = SchedulerOrchestrator() if orchestrator is None else orchestrator

        # Get multi-task timeout from config
        # 读取批量并发任务执行的超时时间。
        # 如果没有 config，则默认为 None，由 ThreadManager 或调用方进一步决定。
        self.multi_task_running_timeout = (
            self.config.get("multi_task_running_timeout") if self.config else None
        )

        # Only initialize thread pool if in parallel mode
        # 保存是否启用并行 dispatch。
        # 开启时 handler 会在线程池中异步执行；关闭时会同步执行。
        self.enable_parallel_dispatch = enable_parallel_dispatch

        # dispatcher 线程名前缀，便于日志和调试区分线程来源。
        self.thread_name_prefix = "dispatcher"

        # 并行模式下创建上下文感知线程池。
        # ContextThreadPoolExecutor 能把请求上下文传递到工作线程中。
        if self.enable_parallel_dispatch:
            self.dispatcher_executor = ContextThreadPoolExecutor(
                max_workers=self.max_workers, thread_name_prefix=self.thread_name_prefix
            )
            # 记录线程池 worker 数。
            logger.info(f"Max works of dispatcher is set to {self.max_workers}")

        # 非并行模式不创建线程池。
        # 后续 execute_task 会直接调用 wrapped_handler。
        else:
            self.dispatcher_executor = None

        # 记录当前是否启用并行分发。
        logger.info(f"enable_parallel_dispatch is set to {self.enable_parallel_dispatch}")

        # Registered message handlers
        # handler 注册表：key 是 task label，value 是处理该 label 的函数。
        self.handlers: dict[str, Callable] = {}

        # Dispatcher running state
        # dispatcher 运行状态标记。
        # 当前类中主要由上下文管理器和 shutdown 修改。
        self._running = False

        # Set to track active futures for monitoring purposes
        # 保存已提交到线程池但尚未完成或尚未清理的 future。
        # stats、join、shutdown 都会参考它。
        self._futures = set()

        # Thread race module for competitive task execution
        # ThreadManager 复用 dispatcher 的线程池，提供竞争式和多任务并发执行能力。
        self.thread_manager = ThreadManager(thread_pool_executor=self.dispatcher_executor)

        # Task tracking for monitoring
        # 正在执行的任务表。
        # key 是 RunningTaskItem.item_id，value 是 RunningTaskItem。
        self._running_tasks: dict[str, RunningTaskItem] = {}

        # 锁用于保护 _running_tasks 和 _futures 等跨线程共享结构。
        self._task_lock = threading.Lock()

        # Configure shutdown wait behavior from config or default
        # 关闭线程池时是否等待任务结束。
        # config 中没有 stop_wait 时使用 DEFAULT_STOP_WAIT。
        self.stop_wait = (
            self.config.get("stop_wait", DEFAULT_STOP_WAIT) if self.config else DEFAULT_STOP_WAIT
        )

        # metrics 用于记录任务等待时间、执行时间、成功/失败计数等。
        self.metrics = metrics

        # status_tracker 用于记录任务生命周期状态。
        self.status_tracker = status_tracker

        # Web 日志提交回调。
        # 当某个业务 task_id 下所有 item 完成/失败时，_maybe_emit_task_completion 会调用它。
        self.submit_web_logs = submit_web_logs  # ADDED

    # 队列收到消息后的回调入口。
    # 目前这个方法不做实际处理，因为相关逻辑已经移到 BaseScheduler 中。
    def on_messages_enqueued(self, msgs: list[ScheduleMessageItem]) -> None:
        # 空消息无需处理。
        if not msgs:
            return
        # This is handled in BaseScheduler now

    # 为真实 handler 创建一层包装函数。
    # 包装函数负责状态追踪、上下文传播、metrics、monitor event、异常记录、Redis ack 和 running task 清理。
    def _create_task_wrapper(self, handler: Callable, task_item: RunningTaskItem):
        """
        Create a wrapper around the handler to track task execution and capture results.

        Args:
            handler: The original handler function
            task_item: The RunningTaskItem to track

        Returns:
            Wrapped handler function that captures results and logs completion
        """

        # wrapped_handler 是实际被线程池或同步路径执行的函数。
        # 它接收同一 user/cube/label 下的一批 ScheduleMessageItem。
        def wrapped_handler(messages: list[ScheduleMessageItem]):
            # 记录任务开始执行的 epoch 时间。
            start_time = time.time()

            # 转成 UTC ISO 字符串，供 start/finish monitor event 使用。
            start_iso = datetime.fromtimestamp(start_time, tz=timezone.utc).isoformat()

            # 如果配置了 status_tracker，则逐条标记为 started。
            # 一批消息可能对应多个 item_id，因此需要逐条更新。
            if self.status_tracker:
                for msg in messages:
                    self.status_tracker.task_started(task_id=msg.item_id, user_id=msg.user_id)

            # try 包住真实 handler 执行。
            # 成功时记录完成；失败时记录失败；finally 中无论成功失败都尝试 Redis ack。
            try:
                # 取批次第一条消息作为代表消息。
                # 当前 dispatcher 分组保证同一批通常具有相同 user、cube、label。
                first_msg = messages[0]

                # 获取 trace_id。
                # 如果消息里没有 trace_id，则生成一个新的，保证日志仍有链路 ID。
                trace_id = getattr(first_msg, "trace_id", None) or generate_trace_id()

                # Propagate trace_id and user info to logging context for this handler execution
                # 构造请求上下文，让 handler 内部日志能拿到 trace_id/api_path/user_name。
                ctx = RequestContext(
                    trace_id=trace_id,
                    api_path=getattr(first_msg, "api_path", None),
                    user_name=getattr(first_msg, "user_name", None),
                    user_type=None,
                )

                # 将上下文设置到当前执行线程。
                set_request_context(ctx)

                # --- mark start: record queuing time(now - enqueue_ts)---
                # 当前时间用于计算从 enqueue 到 handler start 的等待时长。
                now = time.time()

                # 用第一条消息代表整个 batch 的排队信息。
                m = first_msg  # All messages in this batch have same user and type

                # 读取消息入队时间。
                enq_ts = getattr(first_msg, "timestamp", None)

                # Path 1: epoch seconds (preferred)
                # 如果入队时间本身是 epoch 秒，直接转成 float。
                if isinstance(enq_ts, int | float):
                    enq_epoch = float(enq_ts)

                # Path 2: datetime -> normalize to UTC epoch
                # 如果入队时间是 datetime，则统一转为 UTC epoch 秒。
                elif hasattr(enq_ts, "timestamp"):
                    dt = enq_ts

                    # naive datetime 没有时区信息时按 UTC 处理，避免本地时区造成 +8h 等偏差。
                    if dt.tzinfo is None:
                        # treat naive as UTC to neutralize +8h skew
                        dt = dt.replace(tzinfo=timezone.utc)

                    # 转成 epoch 秒。
                    enq_epoch = dt.timestamp()

                # 其他情况无法可靠解析，就把入队时间视为当前时间。
                else:
                    # fallback: treat as "just now"
                    enq_epoch = now

                # 等待时长不能为负，因此用 max 截断。
                wait_sec = max(0.0, now - enq_epoch)

                # 将任务排队等待时长写入 metrics。
                self.metrics.observe_task_wait_duration(wait_sec, m.user_id, m.label)

                # 获取消费者 dequeue 阶段写入的 _dequeue_ts。
                # 它用于区分“出队到真正开始执行”的延迟。
                dequeue_ts = getattr(first_msg, "_dequeue_ts", None)

                # start_delay_ms 表示从 dequeue 到 handler start 的时间。
                start_delay_ms = None

                # 只有 _dequeue_ts 是 epoch 秒时才计算。
                if isinstance(dequeue_ts, int | float):
                    start_delay_ms = max(0.0, start_time - dequeue_ts) * 1000

                # 发出 start 监控事件。
                # 这个事件连接 enqueue/dequeue/handler start 三个阶段。
                emit_monitor_event(
                    "start",
                    first_msg,
                    {
                        "start_ts": start_iso,
                        "start_delay_ms": start_delay_ms,
                        "enqueue_ts": to_iso(enq_ts),
                        "dequeue_ts": to_iso(
                            datetime.fromtimestamp(dequeue_ts, tz=timezone.utc)
                            if isinstance(dequeue_ts, int | float)
                            else None
                        ),
                        "event_duration_ms": start_delay_ms,
                        "total_duration_ms": self._calc_total_duration_ms(start_time, enq_ts),
                    },
                )

                # Execute the original handler
                # 调用真正的业务 handler。
                # 这里才是 memory id 后续处理逻辑实际发生的位置。
                result = handler(messages)

                # --- mark done ---
                # 真实 handler 返回后，记录结束时间。
                finish_time = time.time()

                # 计算 handler 执行耗时。
                duration = finish_time - start_time

                # 将执行耗时写入 metrics。
                self.metrics.observe_task_duration(duration, m.user_id, m.label)

                # 如果配置了状态追踪器，则逐条标记任务完成。
                if self.status_tracker:
                    for msg in messages:
                        self.status_tracker.task_completed(task_id=msg.item_id, user_id=msg.user_id)

                    # 检查业务 task_id 下所有 item 是否完成，并可能提交 Web 状态日志。
                    self._maybe_emit_task_completion(messages)

                # metrics 中记录任务完成计数。
                self.metrics.task_completed(user_id=m.user_id, task_type=m.label)

                # 发出 finish 成功监控事件。
                emit_monitor_event(
                    "finish",
                    first_msg,
                    {
                        "status": "ok",
                        "start_ts": start_iso,
                        "finish_ts": datetime.fromtimestamp(
                            finish_time, tz=timezone.utc
                        ).isoformat(),
                        "exec_duration_ms": duration * 1000,
                        "event_duration_ms": duration * 1000,
                        "total_duration_ms": self._calc_total_duration_ms(
                            finish_time, getattr(first_msg, "timestamp", None)
                        ),
                    },
                )
                # Redis ack is handled in finally to cover failure cases

                # Mark task as completed and remove from tracking
                # 成功后更新 RunningTaskItem，并从 running 表中删除。
                with self._task_lock:
                    if task_item.item_id in self._running_tasks:
                        task_item.mark_completed(result)
                        del self._running_tasks[task_item.item_id]

                # 输出 debug 级别完成信息。
                logger.debug(f"Task completed: {task_item.get_execution_info()}")

                # 把真实 handler 的结果返回给 future 或同步调用方。
                return result

            # 捕获真实 handler 或上面状态/metrics 流程中的异常。
            except Exception as e:
                # 取第一条消息作为失败统计代表。
                m = messages[0]

                # 记录失败发生时间。
                finish_time = time.time()

                # metrics 记录任务失败，并带上异常类型。
                self.metrics.task_failed(m.user_id, m.label, type(e).__name__)

                # 如果配置状态追踪器，则逐条标记失败。
                if self.status_tracker:
                    for msg in messages:
                        self.status_tracker.task_failed(
                            task_id=msg.item_id, user_id=msg.user_id, error_message=str(e)
                        )

                    # 可能向 Web 端发出业务 task 失败日志。
                    self._maybe_emit_task_completion(messages, error=e)

                # 发出 finish 失败监控事件。
                emit_monitor_event(
                    "finish",
                    m,
                    {
                        "status": "fail",
                        "start_ts": start_iso,
                        "finish_ts": datetime.fromtimestamp(
                            finish_time, tz=timezone.utc
                        ).isoformat(),
                        "exec_duration_ms": (finish_time - start_time) * 1000,
                        "event_duration_ms": (finish_time - start_time) * 1000,
                        "error_type": type(e).__name__,
                        "error_msg": str(e),
                        "total_duration_ms": self._calc_total_duration_ms(
                            finish_time, getattr(m, "timestamp", None)
                        ),
                    },
                )

                # Mark task as failed and remove from tracking
                # 失败后更新 RunningTaskItem，并从 running 表删除。
                with self._task_lock:
                    if task_item.item_id in self._running_tasks:
                        task_item.mark_failed(str(e))
                        del self._running_tasks[task_item.item_id]

                # 记录任务失败详情。
                logger.error(f"Task failed: {task_item.get_execution_info()}, Error: {e}")

                # 重新抛出异常，让 future 的 done callback 或同步调用方也能感知失败。
                raise

            # finally 用于无论 handler 成功还是失败，都尽量确认 Redis 消息。
            finally:
                # Ensure Redis messages are acknowledged even if handler fails
                # 只有底层队列是 SchedulerRedisQueue 时才需要 ack。
                if (
                    isinstance(self.memos_message_queue, SchedulerRedisQueue)
                    and self.memos_message_queue is not None
                ):
                    try:
                        # 一个 batch 内可能有多条 Redis 消息，需要逐条 ack。
                        for msg in messages:
                            # Redis Stream 消息 ID 存在 msg.redis_message_id。
                            redis_message_id = msg.redis_message_id

                            # 调用 Redis 队列的 ack_message。
                            self.memos_message_queue.ack_message(
                                user_id=msg.user_id,
                                mem_cube_id=msg.mem_cube_id,
                                task_label=msg.label,
                                redis_message_id=redis_message_id,
                                message=msg,
                            )

                    # ack 失败只记录 warning。
                    # 因为 handler 已经执行完，再抛 ack 异常可能掩盖真实业务异常。
                    except Exception as ack_err:
                        logger.warning(f"Ack in finally failed: {ack_err}")

        # 返回包装后的 handler，供 execute_task 调用。
        return wrapped_handler

    # 如果一个业务 task_id 下所有 item 都结束，则向 Web 端提交一条聚合状态日志。
    # 这样前端看到的是业务任务完成/失败，而不是每个内部 ScheduleMessageItem 的细粒度状态。
    def _maybe_emit_task_completion(
        self, messages: list[ScheduleMessageItem], error: Exception | None = None
    ) -> None:
        """If all item_ids under a business task are completed, emit a single completion log."""
        # 没有 Web 日志提交回调或没有状态追踪器时，无法判断并提交聚合状态。
        if not self.submit_web_logs or not self.status_tracker:
            return

        # messages in one batch can belong to different business task_ids; check each
        # 收集本批消息涉及的业务 task_id。
        task_ids = set()

        # 记录每个 task_id 对应的 source_doc_id。
        # source_doc_id 通常来自上传文档场景，用于前端关联文档。
        task_id_to_doc_id = {}

        # 遍历消息，提取业务 task_id 和 source_doc_id。
        for msg in messages:
            # task_id 是业务级任务 ID，不同于 msg.item_id。
            tid = getattr(msg, "task_id", None)

            # 有 task_id 才参与聚合状态判断。
            if tid:
                task_ids.add(tid)

                # Try to capture source_doc_id for this task if we haven't already
                # 每个 task_id 只记录第一个可用 source_doc_id。
                if tid not in task_id_to_doc_id:
                    info = msg.info or {}
                    sid = info.get("source_doc_id")
                    if sid:
                        task_id_to_doc_id[tid] = sid

        # 没有业务 task_id 时，不需要提交 Web 状态日志。
        if not task_ids:
            return

        # Use the first message only for shared fields; mem_cube_id is same within a batch
        # 用第一条消息提取共享字段。
        first = messages[0]

        # 业务状态日志需要 user_id。
        user_id = first.user_id

        # 业务状态日志需要 mem_cube_id。
        mem_cube_id = first.mem_cube_id

        try:
            # playground API 不提交此类 Web 状态日志，避免测试/演示请求污染前端任务日志。
            if is_playground_api():
                return

            # 对每个业务 task_id 单独查询聚合状态。
            for task_id in task_ids:
                # 获取该 task_id 对应的文档 ID，如果存在。
                source_doc_id = task_id_to_doc_id.get(task_id)

                # 从状态追踪器查询业务任务聚合状态。
                status_data = self.status_tracker.get_task_status_by_business_id(
                    business_task_id=task_id, user_id=user_id
                )

                # 状态追踪器无数据时跳过。
                if not status_data:
                    continue

                # 取聚合状态。
                status = status_data.get("status")

                # 如果聚合状态是 completed，则提交完成日志。
                if status == "completed":
                    # Only emit success log if we didn't just catch an exception locally
                    # (Although if status is 'completed', local error shouldn't happen theoretically,
                    # unless status update lags or is inconsistent. We trust status_tracker here.)
                    # 构造 Web 端可显示的任务完成日志。
                    event = ScheduleLogForWebItem(
                        task_id=task_id,
                        user_id=user_id,
                        mem_cube_id=mem_cube_id,
                        label="taskStatus",
                        from_memory_type="status",
                        to_memory_type="status",
                        log_content=f"Task {task_id} completed",
                        status="completed",
                        source_doc_id=source_doc_id,
                        api_path=getattr(messages[0], "api_path", None) if messages else None,
                    )

                    # 提交完成日志。
                    self.submit_web_logs(event)

                # 如果聚合状态是 failed，则提交失败日志。
                elif status == "failed":
                    # Construct error message
                    # 优先使用当前捕获到的本地异常。
                    error_msg = str(error) if error else None

                    # 如果没有本地异常，则尝试从 status_tracker 聚合错误中取。
                    if not error_msg:
                        # Try to get errors from status_tracker aggregation
                        errors = status_data.get("errors", [])
                        if errors:
                            error_msg = "; ".join(errors)
                        else:
                            error_msg = "Unknown error (check system logs)"

                    # 构造 Web 端失败日志。
                    event = ScheduleLogForWebItem(
                        task_id=task_id,
                        user_id=user_id,
                        mem_cube_id=mem_cube_id,
                        label="taskStatus",
                        from_memory_type="status",
                        to_memory_type="status",
                        log_content=f"Task {task_id} failed: {error_msg}",
                        status="failed",
                        source_doc_id=source_doc_id,
                        api_path=getattr(messages[0], "api_path", None) if messages else None,
                    )

                    # 提交失败日志。
                    self.submit_web_logs(event)

        # 聚合状态日志失败不应该影响业务 handler 结果。
        except Exception:
            logger.warning(
                "Failed to emit task completion log. user_id=%s mem_cube_id=%s task_ids=%s",
                user_id,
                mem_cube_id,
                list(task_ids),
                exc_info=True,
            )

    # 获取当前正在运行的任务。
    # 可选 filter_func 用于按用户、任务名、状态等条件过滤。
    def get_running_tasks(
        self, filter_func: Callable[[RunningTaskItem], bool] | None = None
    ) -> dict[str, RunningTaskItem]:
        """
        Get a copy of currently running tasks, optionally filtered by a custom function.

        Args:
            filter_func: Optional function that takes a RunningTaskItem and returns True if it should be included.
                        Common filters can be created using helper methods like filter_by_user_id, filter_by_task_name, etc.

        Returns:
            Dictionary of running tasks keyed by task ID

        Examples:
            # Get all running tasks
            all_tasks = dispatcher.get_running_tasks()

            # Get tasks for specific user
            user_tasks = dispatcher.get_running_tasks(lambda task: task.user_id == "user123")

            # Get tasks for specific task name
            handler_tasks = dispatcher.get_running_tasks(lambda task: task.task_name == "test_handler")

            # Get tasks with multiple conditions
            filtered_tasks = dispatcher.get_running_tasks(
                lambda task: task.user_id == "user123" and task.status == "running"
            )
        """
        # 加锁读取 _running_tasks，避免与 execute_task/wrapped_handler 的写操作并发冲突。
        with self._task_lock:
            # 没有过滤函数时，返回浅拷贝，避免调用方直接修改内部字典。
            if filter_func is None:
                return self._running_tasks.copy()

            # 有过滤函数时，只返回满足条件的任务。
            return {
                task_id: task_item
                for task_id, task_item in self._running_tasks.items()
                if filter_func(task_item)
            }

    # 获取当前运行中的任务数量。
    def get_running_task_count(self) -> int:
        """
        Get the count of currently running tasks.

        Returns:
            Number of running tasks
        """
        # 加锁读取，保证数量与写入操作互斥。
        with self._task_lock:
            return len(self._running_tasks)

    # 注册单个 label 对应的 handler。
    # dispatcher.dispatch 时会根据消息 label 找到这里注册的 handler。
    def register_handler(
        self,
        label: str,
        handler: Callable[[list[ScheduleMessageItem]], None],
        priority: TaskPriorityLevel | None = None,
        min_idle_ms: int | None = None,
    ):
        """
        Register a handler function for a specific message label.

        Args:
            label: Message label to handle
            handler: Callable that processes messages of this label
            priority: Optional priority level for the task
            min_idle_ms: Optional minimum idle time for task claiming
        """
        # 将 label -> handler 写入注册表。
        self.handlers[label] = handler

        # 如果有 orchestrator，也同步写入该 label 的调度配置。
        # 后续 BaseSchedulerQueueMixin.submit_messages 可以通过 orchestrator 查询任务优先级。
        if self.orchestrator:
            self.orchestrator.set_task_config(
                task_label=label, priority=priority, min_idle_ms=min_idle_ms
            )

    # 批量注册多个 handler。
    # 支持两种 value：直接传 handler，或传 (handler, priority, min_idle_ms) 元组。
    def register_handlers(
        self,
        handlers: dict[
            str,
            Callable[[list[ScheduleMessageItem]], None]
            | tuple[
                Callable[[list[ScheduleMessageItem]], None], TaskPriorityLevel | None, int | None
            ],
        ],
    ) -> None:
        """
        Bulk register multiple handlers from a dictionary.

        Args:
            handlers: Dictionary where key is label and value is either:
                     - handler_callable
                     - tuple(handler_callable, priority, min_idle_ms)
        """
        # 遍历调用方提供的所有 label/handler 配置。
        for label, value in handlers.items():
            # label 必须是字符串，否则无法与 ScheduleMessageItem.label 正确匹配。
            if not isinstance(label, str):
                logger.error(f"Invalid label type: {type(label)}. Expected str.")
                continue

            # 如果 value 是 tuple，则解析为 handler、priority、min_idle_ms。
            if isinstance(value, tuple):
                # tuple 必须严格包含 3 个元素。
                if len(value) != 3:
                    logger.error(
                        f"Invalid handler tuple for label '{label}'. Expected (handler, priority, min_idle_ms)."
                    )
                    continue

                # 解包 handler 和调度配置。
                handler, priority, min_idle_ms = value

            # 如果 value 不是 tuple，则只把它当作 handler，优先级配置使用默认值。
            else:
                handler = value
                priority = None
                min_idle_ms = None

            # handler 必须可调用。
            if not callable(handler):
                logger.error(f"Handler for label '{label}' is not callable.")
                continue

            # 调用单个注册方法完成注册和 orchestrator 配置。
            self.register_handler(
                label=label, handler=handler, priority=priority, min_idle_ms=min_idle_ms
            )

        # 记录批量注册数量。
        logger.info(f"Registered {len(handlers)} handlers in bulk")

    # 注销单个 label 的 handler。
    def unregister_handler(self, label: str) -> bool:
        """
        Unregister a handler for a specific label.

        Args:
            label: The label to unregister the handler for

        Returns:
            bool: True if handler was found and removed, False otherwise
        """
        # 如果 label 已注册，则删除 handler。
        if label in self.handlers:
            del self.handlers[label]

            # 同步删除 orchestrator 中的任务配置。
            if self.orchestrator:
                self.orchestrator.remove_task_config(label)

            # 记录注销成功。
            logger.info(f"Unregistered handler for label: {label}")
            return True

        # label 未注册时返回 False。
        else:
            logger.warning(f"No handler found for label: {label}")
            return False

    # 批量注销多个 label 的 handler。
    def unregister_handlers(self, labels: list[str]) -> dict[str, bool]:
        """
        Unregister multiple handlers by their labels.

        Args:
            labels: List of labels to unregister handlers for

        Returns:
            dict[str, bool]: Dictionary mapping each label to whether it was successfully unregistered
        """
        # 保存每个 label 的注销结果。
        results = {}

        # 逐个注销。
        for label in labels:
            results[label] = self.unregister_handler(label)

        # 记录批量注销数量。
        logger.info(f"Unregistered handlers for {len(labels)} labels")

        # 返回每个 label 是否注销成功。
        return results

    # 返回 dispatcher 运行时轻量统计信息。
    def stats(self) -> dict[str, int]:
        """
        Lightweight runtime stats for monitoring.

        Returns:
            {
                'running': <number of running tasks>,
                'inflight': <number of futures tracked (pending+running)>,
                'handlers': <registered handler count>,
            }
        """
        try:
            # 正在执行的任务数量。
            running = self.get_running_task_count()
        except Exception:
            running = 0

        try:
            # 清理已经完成的 future，只保留仍在飞行中的 future。
            with self._task_lock:
                done = {f for f in self._futures if f.done()}
                if done:
                    self._futures -= done
                inflight = len(self._futures)
        except Exception:
            inflight = 0

        try:
            # 当前注册的 handler 数量。
            handlers = len(self.handlers)
        except Exception:
            handlers = 0

        # 返回统一统计字典。
        return {"running": running, "inflight": inflight, "handlers": handlers}

    # 默认消息处理函数。
    # 如果某个 label 没有注册 handler，dispatcher 会使用它兜底。
    def _default_message_handler(self, messages: list[ScheduleMessageItem]) -> None:
        logger.debug(f"Using _default_message_handler to deal with messages: {messages}")

    # 线程池 future 完成后的回调。
    # 它负责从 _futures 集合中移除 future，并读取结果以暴露 handler 异常。
    def _handle_future_result(self, future):
        # 先从 inflight 集合中移除。
        with self._task_lock:
            self._futures.discard(future)

        try:
            # 调用 future.result() 会重新抛出线程内异常。
            future.result()  # this will throw exception
        except Exception as e:
            # 记录 handler 执行失败。
            logger.error(f"Handler execution failed: {e!s}", exc_info=True)

    # 计算从入队到完成的总耗时，单位毫秒。
    @staticmethod
    def _calc_total_duration_ms(finish_epoch: float, enqueue_ts) -> float | None:
        """
        Calculate total duration from enqueue timestamp to finish time in milliseconds.
        """
        try:
            # enq_epoch 保存入队时间的 epoch 秒数。
            enq_epoch = None

            # enqueue_ts 是 epoch 秒时直接使用。
            if isinstance(enqueue_ts, int | float):
                enq_epoch = float(enqueue_ts)

            # enqueue_ts 是 datetime 时，转成 UTC epoch 秒。
            elif hasattr(enqueue_ts, "timestamp"):
                dt = enqueue_ts

                # naive datetime 按 UTC 处理。
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)

                # 转为 epoch 秒。
                enq_epoch = dt.timestamp()

            # 无法解析入队时间时返回 None。
            if enq_epoch is None:
                return None

            # 总耗时 = 完成时间 - 入队时间。
            total_ms = max(0.0, finish_epoch - enq_epoch) * 1000
            return total_ms

        # 任意异常都返回 None，避免监控字段计算影响主流程。
        except Exception:
            return None

    # 执行一个分组后的任务。
    # 这里会创建 RunningTaskItem、注册 running 状态、包装 handler，并根据并行配置决定线程池执行或同步执行。
    def execute_task(
        self,
        user_id: str,
        mem_cube_id: str,
        task_label: str,
        msgs: list[ScheduleMessageItem],
        handler_call_back: Callable[[list[ScheduleMessageItem]], Any],
    ):
        # 兼容调用方传入单条 ScheduleMessageItem。
        if isinstance(msgs, ScheduleMessageItem):
            msgs = [msgs]

        # Create task tracking item for this dispatch
        # 构造运行中任务记录。
        # 注意这是 dispatcher 层的一次 dispatch 任务，不等同于每条 ScheduleMessageItem 的 item_id。
        task_item = RunningTaskItem(
            user_id=user_id,
            mem_cube_id=mem_cube_id,
            task_info=f"Processing {len(msgs)} message(s) with label '{task_label}' for user {user_id} and mem_cube {mem_cube_id}",
            task_name=f"{task_label}_handler",
            messages=msgs,
        )

        # Uniformly register the task before execution
        # 在真正执行前把任务放入 running 表。
        # 这样即使马上开始执行，监控也能看到它。
        with self._task_lock:
            self._running_tasks[task_item.item_id] = task_item

        # Create wrapped handler for task tracking
        # 创建带状态追踪/监控/ack 的包装 handler。
        wrapped_handler = self._create_task_wrapper(handler_call_back, task_item)

        # dispatch to different handler
        # 记录任务开始调度。
        logger.debug(f"Task started: {task_item.get_execution_info()}")

        # If priority is LEVEL_1, force synchronous execution regardless of thread pool availability
        # 当前代码实际只根据 enable_parallel_dispatch 和 dispatcher_executor 判断是否用线程池。
        # 注释提到 LEVEL_1，但 execute_task 本身没有直接接收 priority 参数。
        use_thread_pool = self.enable_parallel_dispatch and self.dispatcher_executor is not None

        # 并行模式：把 wrapped_handler 提交到线程池。
        if use_thread_pool:
            # Submit and track the future
            # 提交任务到线程池，返回 future。
            future = self.dispatcher_executor.submit(wrapped_handler, msgs)

            # 将 future 加入 inflight 集合，供 stats/join/shutdown 追踪。
            with self._task_lock:
                self._futures.add(future)

            # 注册完成回调，任务结束后清理 future 并记录异常。
            future.add_done_callback(self._handle_future_result)

            # 记录分发日志。
            logger.debug(
                f"Dispatch {len(msgs)} message(s) to {task_label} handler for user {user_id} and mem_cube {mem_cube_id}."
            )

        # 同步模式：直接在当前线程执行 wrapped_handler。
        else:
            # For synchronous execution, the wrapper will run and remove the task upon completion
            # 记录同步执行日志。
            logger.debug(
                f"Execute {len(msgs)} message(s) synchronously for {task_label} for user {user_id} and mem_cube {mem_cube_id}."
            )

            # 直接执行包装 handler。
            wrapped_handler(msgs)

    # 将一批消息按 user_id、mem_cube_id 和 label 分组，然后派发到对应 handler。
    def dispatch(self, msg_list: list[ScheduleMessageItem]):
        """
        Dispatch a list of messages to their respective handlers.

        Args:
            msg_list: List of ScheduleMessageItem objects to process
        """
        # 空列表无需分发。
        if not msg_list:
            logger.debug("Received empty message list, skipping dispatch")
            return

        # Group messages by user_id and mem_cube_id first
        # 第一层分组：按 user_id 和 mem_cube_id 组织消息。
        # 这样可以保证同一 handler 调用处理的是同一用户、同一 cube 下的消息。
        user_cube_groups = group_messages_by_user_and_mem_cube(msg_list)

        # 记录批次规模、用户组数量和 label 种类。
        logger.info(
            "Dispatcher received batch. total_messages=%s user_groups=%s unique_labels=%s",
            len(msg_list),
            len(user_cube_groups),
            sorted({msg.label for msg in msg_list}),
        )

        # Process each user and mem_cube combination
        # 遍历每个 user_id。
        for user_id, cube_groups in user_cube_groups.items():
            # 遍历该用户下每个 mem_cube_id。
            for mem_cube_id, user_cube_msgs in cube_groups.items():
                # Group messages by their labels within each user/mem_cube combination
                # 第二层分组：在同一 user/cube 下按 label 分组。
                # 同一 label 的消息会合并成一个 batch 交给同一个 handler。
                label_groups = defaultdict(list)

                # 把消息放入对应 label 的列表。
                for message in user_cube_msgs:
                    label_groups[message.label].append(message)

                # Process each label group within this user/mem_cube combination
                # 对每个 label 分组分别执行任务。
                for label, msgs in label_groups.items():
                    # 根据 label 找到注册 handler。
                    # 如果没注册，则使用默认 handler 兜底。
                    handler = self.handlers.get(label, self._default_message_handler)

                    # 通过 execute_task 统一执行，保留任务追踪、包装、线程池、监控等逻辑。
                    self.execute_task(
                        user_id=user_id,
                        mem_cube_id=mem_cube_id,
                        task_label=label,
                        msgs=msgs,
                        handler_call_back=handler,
                    )

    # 等待所有已分发到线程池的任务完成。
    def join(self, timeout: float | None = None) -> bool:
        """Wait for all dispatched tasks to complete.

        Args:
            timeout: Maximum time to wait in seconds. None means wait forever.

        Returns:
            bool: True if all tasks completed, False if timeout occurred.
        """
        # 非并行模式没有线程池任务需要等待。
        if not self.enable_parallel_dispatch or self.dispatcher_executor is None:
            return True  # Serial mode requires no waiting

        # 等待当前记录的 futures 全部完成，或者直到 timeout。
        done, not_done = concurrent.futures.wait(
            self._futures, timeout=timeout, return_when=concurrent.futures.ALL_COMPLETED
        )

        # Check for exceptions in completed tasks
        # 对已经完成的 future 调用 result，确保异常被记录。
        for future in done:
            try:
                future.result()
            except Exception:
                logger.error("Handler failed during shutdown", exc_info=True)

        # 如果没有未完成 future，表示全部任务完成。
        return len(not_done) == 0

    # 竞争式执行多个任务，返回最先完成的任务结果。
    def run_competitive_tasks(
        self, tasks: dict[str, Callable[[threading.Event], Any]], timeout: float = 10.0
    ) -> tuple[str, Any] | None:
        """
        Run multiple tasks in a competitive race, returning the result of the first task to complete.

        Args:
            tasks: Dictionary mapping task names to task functions that accept a stop_flag parameter
            timeout: Maximum time to wait for any task to complete (in seconds)

        Returns:
            Tuple of (task_name, result) from the winning task, or None if no task completes
        """
        # 记录竞争任务数量。
        logger.info(f"Starting competitive execution of {len(tasks)} tasks")

        # 交给 ThreadManager 执行竞争逻辑。
        return self.thread_manager.run_race(tasks, timeout)

    # 并发执行多个命名任务，并返回所有任务结果。
    def run_multiple_tasks(
        self,
        tasks: dict[str, tuple[Callable, tuple]],
        use_thread_pool: bool | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """
        Execute multiple tasks concurrently and return all results.

        Args:
            tasks: Dictionary mapping task names to (task_execution_function, task_execution_parameters) tuples
            use_thread_pool: Whether to use ThreadPoolExecutor. If None, uses dispatcher's parallel mode setting
            timeout: Maximum time to wait for all tasks to complete (in seconds). If None, uses config default.

        Returns:
            Dictionary mapping task names to their results

        Raises:
            TimeoutError: If tasks don't complete within the specified timeout
        """
        # Use dispatcher's parallel mode setting if not explicitly specified
        # 调用方未指定是否使用线程池时，继承 dispatcher 的并行配置。
        if use_thread_pool is None:
            use_thread_pool = self.enable_parallel_dispatch

        # Use config timeout if not explicitly provided
        # 调用方未传 timeout 时，使用初始化时从 config 读取的默认超时时间。
        if timeout is None:
            timeout = self.multi_task_running_timeout

        # 记录多任务执行参数。
        logger.info(
            f"Executing {len(tasks)} tasks concurrently (thread_pool: {use_thread_pool}, timeout: {timeout})"
        )

        try:
            # 交给 ThreadManager 执行多个任务。
            results = self.thread_manager.run_multiple_tasks(
                tasks=tasks, use_thread_pool=use_thread_pool, timeout=timeout
            )

            # 记录成功完成数量。
            logger.info(
                f"Successfully completed {len([r for r in results.values() if r is not None])}/{len(tasks)} tasks"
            )

            # 返回任务名到结果的映射。
            return results

        # 多任务执行失败时记录并继续向外抛出。
        except Exception as e:
            logger.error(f"Multiple tasks execution failed: {e}", exc_info=True)
            raise

    # 优雅关闭 dispatcher。
    def shutdown(self) -> None:
        """Gracefully shutdown the dispatcher."""
        # 标记 dispatcher 不再运行。
        self._running = False

        # Shutdown executor
        # 尝试关闭线程池。
        try:
            # wait 参数来自配置，cancel_futures=True 会取消尚未开始执行的 future。
            self.dispatcher_executor.shutdown(wait=self.stop_wait, cancel_futures=True)

        # 关闭异常只记录日志。
        except Exception as e:
            logger.error(f"Executor shutdown error: {e}", exc_info=True)

        # 无论关闭是否成功，都清空 future 追踪集合。
        finally:
            self._futures.clear()

    # 上下文管理器入口。
    # 使用 with SchedulerDispatcher(...) as dispatcher 时会设置运行状态。
    def __enter__(self):
        self._running = True
        return self

    # 上下文管理器退出。
    # 自动调用 shutdown 释放线程池资源。
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
