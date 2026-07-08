# 启用 postponed evaluation of annotations。
# 这样类型注解不会在函数定义时立即求值，既可以减少运行期开销，也能避免部分循环引用导入问题。
from __future__ import annotations

# multiprocessing 用于在配置为“进程启动模式”时，把消息消费者放到独立子进程中运行。
# 相比线程，进程可以隔离 GIL 和部分运行状态，但上下文和资源管理也更复杂。
import multiprocessing

# time 用于消费者循环休眠、计算出队时间以及队列等待耗时。
import time

# suppress 用于吞掉非关键异常。
# 例如 metrics 上报失败不应该影响消息入队主流程。
from contextlib import suppress

# datetime/timezone 用于生成带 UTC 时区的 dequeue_ts，并处理 timestamp 的时区归一化。
from datetime import datetime, timezone

# TYPE_CHECKING 用于只在类型检查阶段导入 Callable。
# 这样运行时不会引入额外依赖，也避免潜在循环导入。
from typing import TYPE_CHECKING

# 从请求上下文模块引入线程、上下文对象和上下文读写函数。
# 调度任务从请求线程进入后台线程/进程后，需要显式透传 trace_id、api_path、user_name 等上下文信息。
from memos.context.context import (
    # ContextThread 是项目自定义线程类，通常用于在新线程中保留或传播上下文变量。
    ContextThread,

    # RequestContext 表示一次请求的上下文快照。
    # 消费队列消息时，会用消息中保存的 trace_id/api_path/user_name 重建上下文。
    RequestContext,

    # 获取当前 API path，用于把任务来源接口写入 ScheduleMessageItem。
    get_current_api_path,

    # 获取当前上下文，用于消费者临时切换上下文后再恢复原上下文。
    get_current_context,

    # 获取当前 trace_id，用于关联请求日志、入队日志和任务执行日志。
    get_current_trace_id,

    # 设置当前线程/上下文中的 RequestContext。
    set_request_context,
)

# 获取项目统一 logger。
# 当前 mixin 会记录入队汇总、消费者状态、线程/进程启动停止等关键日志。
from memos.log import get_logger

# STARTUP_BY_PROCESS 是调度器启动方式常量。
# 当 scheduler_startup_mode 等于该常量时，消费者用 multiprocessing.Process 启动。
from memos.mem_scheduler.schemas.general_schemas import STARTUP_BY_PROCESS

# ScheduleMessageItem 是调度系统中的消息模型。
# submit_messages、消费者、dispatcher 都围绕该对象传递任务信息。
from memos.mem_scheduler.schemas.message_schemas import ScheduleMessageItem

# TaskPriorityLevel 表示任务优先级。
# LEVEL_1 在这里被视为即时任务，不进入普通队列，直接交给 dispatcher 执行。
from memos.mem_scheduler.schemas.task_schemas import TaskPriorityLevel

# get_utc_now 用于给缺少 timestamp 的消息补 UTC 时间。
# 这个 timestamp 后面会用于计算 queue_wait_ms。
from memos.mem_scheduler.utils.db_utils import get_utc_now

# group_messages_by_user_and_mem_cube 用于按 user_id 和 mem_cube_id 分组消息。
# 即时任务直接执行时，会先按用户和 cube 分组，再按 label 分组交给 dispatcher。
from memos.mem_scheduler.utils.misc_utils import group_messages_by_user_and_mem_cube

# emit_monitor_event 用于上报 enqueue/dequeue 等监控事件。
# to_iso 用于把 timestamp 统一转成 ISO 字符串。
from memos.mem_scheduler.utils.monitor_event_utils import emit_monitor_event, to_iso


# 当前模块 logger。
# 使用 __name__ 能让日志里显示具体来源模块。
logger = get_logger(__name__)

# 类型检查阶段才导入 Callable，运行时不会执行。
if TYPE_CHECKING:
    # Callable 用于 handlers 属性和 register_handlers 的类型标注。
    from collections.abc import Callable


# BaseSchedulerQueueMixin 提供调度队列相关的通用能力。
# 它假设实际 Scheduler 类已经提供了若干属性：memos_message_queue、dispatcher、metrics、status_tracker、orchestrator 等。
# 因此这个类更像“混入实现”，负责把消息提交、消费、监控和生命周期管理逻辑复用到具体 Scheduler 中。
class BaseSchedulerQueueMixin:
    # 提交一条或多条调度消息。
    # 这里会完成上下文透传、状态追踪、禁用 handler 过滤、优先级拆分，以及即时任务/队列任务的不同处理路径。
    def submit_messages(self, messages: ScheduleMessageItem | list[ScheduleMessageItem]):
        # 允许调用方传单条 ScheduleMessageItem。
        # 为了后续统一遍历处理，单条消息会先包装成列表。
        if isinstance(messages, ScheduleMessageItem):
            messages = [messages]

        # 如果传入空列表或 None 风格的假值，直接返回。
        # 空提交不需要记录错误，因为上层可能合法地没有任务要提交。
        if not messages:
            return

        # 读取当前请求的 trace_id。
        # 提交到后台调度系统后，trace_id 仍可用于关联原始 API 请求和后续任务执行日志。
        current_trace_id = get_current_trace_id()

        # 读取当前请求的 API path。
        # 它会写入消息，便于监控中区分任务是由哪个接口触发。
        current_api_path = get_current_api_path()

        # immediate_msgs 存放高优先级、需要立即执行的消息。
        # 这些消息不会进入底层队列，而是直接按分组交给 dispatcher。
        immediate_msgs: list[ScheduleMessageItem] = []

        # queued_msgs 存放普通优先级消息。
        # 这些消息会交给 self.memos_message_queue.submit_messages 进入 Redis 或本地队列。
        queued_msgs: list[ScheduleMessageItem] = []

        # 遍历每条待提交消息，补齐上下文和状态信息，并按优先级分流。
        for msg in messages:
            # 如果当前上下文有 trace_id，则写入消息。
            # 这里会覆盖消息已有 trace_id，优先使用当前请求链路。
            if current_trace_id:
                msg.trace_id = current_trace_id

            # 如果当前上下文有 API path，且消息没有显式 api_path，则写入当前 API path。
            # 不覆盖已有 api_path 是为了保留上游已经指定的来源。
            if current_api_path and not getattr(msg, "api_path", None):
                msg.api_path = current_api_path

            # 尝试记录“任务已入队/提交”的 metrics。
            # suppress(Exception) 表示 metrics 失败不影响调度主流程。
            with suppress(Exception):
                self.metrics.task_enqueued(user_id=msg.user_id, task_type=msg.label)

            # 如果消息没有 timestamp，则用当前 UTC 时间补齐。
            # 后续 enqueue/dequeue 监控会基于这个时间计算队列等待时长。
            if getattr(msg, "timestamp", None) is None:
                msg.timestamp = get_utc_now()

            # 如果配置了状态追踪器，则记录任务已提交状态。
            # status_tracker 适合持久化任务状态，方便 API 查询任务生命周期。
            if self.status_tracker:
                try:
                    # 将消息中的业务字段转换成状态追踪器需要的字段。
                    self.status_tracker.task_submitted(
                        task_id=msg.item_id,
                        user_id=msg.user_id,
                        task_type=msg.label,
                        mem_cube_id=msg.mem_cube_id,
                        business_task_id=msg.task_id,
                    )
                # 状态追踪失败不能阻断任务提交，只记录 warning。
                except Exception:
                    logger.warning("status_tracker.task_submitted failed", exc_info=True)

            # 如果该消息的 handler label 被禁用，则跳过这条消息。
            # 这通常用于临时关闭某类后台任务，例如调试期间禁用某个 scheduler handler。
            if self.disabled_handlers and msg.label in self.disabled_handlers:
                logger.debug(
                    "Skip disabled handler. label=%s item_id=%s user_id=%s mem_cube_id=%s",
                    msg.label,
                    msg.item_id,
                    msg.user_id,
                    msg.mem_cube_id,
                )
                # continue 表示这条消息既不会即时执行，也不会进入普通队列。
                continue

            # 通过 orchestrator 查询该任务 label 对应的优先级。
            # 优先级决定消息是直接执行，还是进入队列等待消费者拉取。
            task_priority = self.orchestrator.get_task_priority(task_label=msg.label)

            # LEVEL_1 任务被视为最高优先级任务。
            # 它会进入 immediate_msgs，稍后直接调用 dispatcher.execute_task。
            if task_priority == TaskPriorityLevel.LEVEL_1:
                immediate_msgs.append(msg)

            # 其他优先级任务进入普通队列。
            # 后续由 _message_consumer 周期性从队列中拉取并 dispatch。
            else:
                queued_msgs.append(msg)

        # 输出本次提交的汇总信息。
        # total 是传入消息数，immediate/queued 是过滤 disabled handler 后实际分流的数量。
        logger.info(
            "Submit scheduler messages summary. total=%s immediate=%s queued=%s queue_backend=%s",
            len(messages),
            len(immediate_msgs),
            len(queued_msgs),
            "redis_queue" if self.use_redis_queue else "local_queue",
        )

        # 如果存在即时任务，直接在当前调度路径中执行。
        # 这类任务不经过队列，因此需要手动补发 enqueue/dequeue 监控事件。
        if immediate_msgs:
            # 先为每个即时任务发 enqueue 事件。
            # 虽然它不进入真实队列，但为了监控口径一致，仍记录一次“入队”。
            for m in immediate_msgs:
                emit_monitor_event(
                    "enqueue",
                    m,
                    {
                        "enqueue_ts": to_iso(getattr(m, "timestamp", None)),
                        "event_duration_ms": 0,
                        "total_duration_ms": 0,
                    },
                )

            # 再为每个即时任务发 dequeue 事件。
            # 对即时任务来说，出队时间几乎等于当前时间，queue_wait_ms 代表从 timestamp 到现在的等待时间。
            for m in immediate_msgs:
                try:
                    # 当前时间戳，单位秒。
                    now = time.time()

                    # 取消息入队时间对象。
                    enqueue_ts_obj = getattr(m, "timestamp", None)

                    # enqueue_epoch 是入队时间的 epoch 秒数。
                    enqueue_epoch = None

                    # 如果 timestamp 本身是 int/float，则直接当作 epoch 秒。
                    if isinstance(enqueue_ts_obj, int | float):
                        enqueue_epoch = float(enqueue_ts_obj)

                    # 如果 timestamp 是 datetime 一类带 timestamp() 方法的对象，则转成 epoch 秒。
                    elif hasattr(enqueue_ts_obj, "timestamp"):
                        dt = enqueue_ts_obj

                        # 如果 datetime 没有时区信息，按 UTC 处理。
                        # 这样可以避免本地时区影响队列等待时间计算。
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=timezone.utc)

                        # 转换成 epoch 秒。
                        enqueue_epoch = dt.timestamp()

                    # 默认无法计算等待时间。
                    queue_wait_ms = None

                    # 如果成功得到入队 epoch，则计算从入队到当前的等待时间。
                    if enqueue_epoch is not None:
                        queue_wait_ms = max(0.0, now - enqueue_epoch) * 1000

                    # 给消息对象附加内部字段 _dequeue_ts。
                    # 使用 object.__setattr__ 是为了兼容可能为 frozen/Pydantic 风格的消息对象。
                    object.__setattr__(m, "_dequeue_ts", now)

                    # 发出即时任务的 dequeue 监控事件。
                    emit_monitor_event(
                        "dequeue",
                        m,
                        {
                            "enqueue_ts": to_iso(enqueue_ts_obj),
                            "dequeue_ts": datetime.fromtimestamp(now, tz=timezone.utc).isoformat(),
                            "queue_wait_ms": queue_wait_ms,
                            "event_duration_ms": queue_wait_ms,
                            "total_duration_ms": queue_wait_ms,
                        },
                    )

                    # 更新 metrics 中的出队计数。
                    self.metrics.task_dequeued(user_id=m.user_id, task_type=m.label)

                # 即时任务监控事件失败不影响后续执行。
                except Exception:
                    logger.debug("Failed to emit dequeue for immediate task", exc_info=True)

            # 将即时任务按 user_id 和 mem_cube_id 分组。
            # dispatcher.execute_task 的语义是按用户/cube/label 维度执行一组消息。
            user_cube_groups = group_messages_by_user_and_mem_cube(immediate_msgs)

            # 遍历每个用户分组。
            for user_id, cube_groups in user_cube_groups.items():
                # 遍历该用户下的每个 memory cube 分组。
                for mem_cube_id, user_cube_msgs in cube_groups.items():
                    # label_groups 用于把同一 user/cube 下的消息再按任务 label 分组。
                    label_groups: dict[str, list[ScheduleMessageItem]] = {}

                    # 将消息归入各自 label。
                    for m in user_cube_msgs:
                        label_groups.setdefault(m.label, []).append(m)

                    # 对每个 label 分组找到对应 handler，并直接交给 dispatcher 执行。
                    for label, msgs_by_label in label_groups.items():
                        # 优先取注册的 handler；找不到时使用 dispatcher 默认 handler。
                        handler = self.dispatcher.handlers.get(
                            label, self.dispatcher._default_message_handler
                        )

                        # 直接执行任务。
                        # 这里绕过队列，所以不会再经过 _message_consumer 的普通 dispatch 路径。
                        self.dispatcher.execute_task(
                            user_id=user_id,
                            mem_cube_id=mem_cube_id,
                            task_label=label,
                            msgs=msgs_by_label,
                            handler_call_back=handler,
                        )

        # 如果存在普通队列任务，则提交到底层队列包装器。
        # 底层可能是 Redis 队列，也可能是本地内存队列。
        if queued_msgs:
            self.memos_message_queue.submit_messages(messages=queued_msgs)

    # 消息消费者主循环。
    # 它负责周期性从底层队列拉取消息，补充出队监控，再交给 dispatcher 分发执行。
    def _message_consumer(self) -> None:
        # 只要 scheduler 处于运行状态，就持续消费。
        while self._running:
            try:
                # 如果启用并行 dispatch，并且 dispatcher 存在，就先检查当前运行任务数。
                if self.enable_parallel_dispatch and self.dispatcher:
                    # 获取当前 dispatcher 正在运行的任务数量。
                    running_tasks = self.dispatcher.get_running_task_count()

                    # 如果运行任务数已经达到线程池上限，就先等待一个消费间隔。
                    # 这是一种背压机制，避免无限拉取消息导致内存堆积。
                    if running_tasks >= self.dispatcher.max_workers:
                        time.sleep(self._consume_interval)
                        continue

                # 从底层队列拉取一批消息。
                # consume_batch 控制单轮最多消费多少条。
                messages = self.memos_message_queue.get_messages(batch_size=self.consume_batch)

                # 如果本轮拿到了消息，则进行出队监控和 dispatch。
                if messages:
                    # 当前出队时间，所有本批消息共享这个 now。
                    now = time.time()

                    # 逐条消息恢复上下文并发出 dequeue 事件。
                    for msg in messages:
                        # 保存消费线程当前上下文，处理完这条消息后恢复。
                        prev_context = get_current_context()
                        try:
                            # 用消息中保存的 trace_id/api_path/user_name 重建请求上下文。
                            # 这样后续日志和监控仍能关联到原始请求。
                            msg_context = RequestContext(
                                trace_id=msg.trace_id,
                                api_path=msg.api_path,
                                user_name=msg.user_name,
                            )

                            # 设置当前上下文为消息上下文。
                            set_request_context(msg_context)

                            # 取消息入队时间。
                            enqueue_ts_obj = getattr(msg, "timestamp", None)

                            # enqueue_epoch 用于计算队列等待时长。
                            enqueue_epoch = None

                            # 数字型 timestamp 直接视为 epoch 秒。
                            if isinstance(enqueue_ts_obj, int | float):
                                enqueue_epoch = float(enqueue_ts_obj)

                            # datetime 型 timestamp 转成 epoch 秒。
                            elif hasattr(enqueue_ts_obj, "timestamp"):
                                dt = enqueue_ts_obj

                                # 没有时区时按 UTC 补齐。
                                if dt.tzinfo is None:
                                    dt = dt.replace(tzinfo=timezone.utc)

                                # 转换成 epoch 秒。
                                enqueue_epoch = dt.timestamp()

                            # 默认无法计算等待耗时。
                            queue_wait_ms = None

                            # 如果有入队 epoch，则计算等待时间。
                            if enqueue_epoch is not None:
                                queue_wait_ms = max(0.0, now - enqueue_epoch) * 1000

                            # 在消息对象上记录出队时间。
                            object.__setattr__(msg, "_dequeue_ts", now)

                            # 发出 dequeue 监控事件。
                            emit_monitor_event(
                                "dequeue",
                                msg,
                                {
                                    "enqueue_ts": to_iso(enqueue_ts_obj),
                                    "dequeue_ts": datetime.fromtimestamp(
                                        now, tz=timezone.utc
                                    ).isoformat(),
                                    "queue_wait_ms": queue_wait_ms,
                                    "event_duration_ms": queue_wait_ms,
                                    "total_duration_ms": queue_wait_ms,
                                },
                            )

                            # 更新出队 metrics。
                            self.metrics.task_dequeued(user_id=msg.user_id, task_type=msg.label)

                        # 无论当前消息处理是否异常，都恢复消费者线程之前的上下文。
                        finally:
                            set_request_context(prev_context)

                    try:
                        # 通知 dispatcher 有消息被取出/准备分发。
                        # suppress 避免该通知失败影响真正 dispatch。
                        with suppress(Exception):
                            if messages:
                                self.dispatcher.on_messages_enqueued(messages)

                        # 如果本轮拉满了 consume_batch，说明队列可能比较忙，输出 debug 级别批次信息。
                        if len(messages) >= self.consume_batch:
                            # 统计本批消息包含的任务 label。
                            unique_labels = sorted({msg.label for msg in messages})

                            # 记录批量消费情况和队列后端。
                            logger.debug(
                                "Consumer dequeued batch. batch_size=%s consume_batch=%s unique_labels=%s queue_backend=%s",
                                len(messages),
                                self.consume_batch,
                                unique_labels,
                                "redis_queue" if self.use_redis_queue else "local_queue",
                            )

                        # 将本批消息交给 dispatcher 统一分发。
                        # dispatcher 内部会按用户、cube、label、handler 等维度执行任务。
                        self.dispatcher.dispatch(messages)

                    # dispatch 失败记录 error，但消费者循环继续运行。
                    except Exception as e:
                        logger.error("Error dispatching messages: %s", e)

                # 每轮消费后休眠固定间隔，避免空轮询占用 CPU。
                time.sleep(self._consume_interval)

            # 捕获消费者循环最外层异常，保证消费者不会因为单次异常退出。
            except Exception as e:
                # Redis 队列空消息异常是预期情况，不需要反复打印 error。
                if "No messages available in Redis queue" not in str(e):
                    logger.error("Unexpected error in message consumer: %s", e, exc_info=True)

                # 异常后也休眠，避免快速失败循环刷屏。
                time.sleep(self._consume_interval)

    # 队列 metrics 监控循环。
    # 它周期性读取队列长度，并把每个 stream/user 的队列长度写入 metrics。
    def _monitor_loop(self):
        # 只要 scheduler 运行，就持续监控。
        while self._running:
            try:
                # 读取底层队列大小。
                # Redis 队列通常返回 dict，包含每个 stream 的长度和 total_size。
                q_sizes = self.memos_message_queue.qsize()

                # 如果返回值不是 dict，说明该队列实现不提供分 stream 统计，跳过本轮。
                if not isinstance(q_sizes, dict):
                    continue

                # 遍历每个 stream 的长度。
                for stream_key, queue_length in q_sizes.items():
                    # total_size 是整体汇总，不对应具体用户，不按用户写 metrics。
                    if stream_key == "total_size":
                        continue

                    # Redis/local stream key 通常由冒号分隔，尾部包含 user_id、mem_cube_id、task_label 等信息。
                    parts = stream_key.split(":")

                    # 当 key 至少有 3 段时，按当前约定取倒数第三段作为 user_id。
                    if len(parts) >= 3:
                        user_id = parts[-3]
                        self.metrics.update_queue_length(queue_length, user_id)

                    # 如果没有冒号，可能是简单本地队列 key，则直接用 stream_key 作为用户或队列标识。
                    else:
                        if ":" not in stream_key:
                            self.metrics.update_queue_length(queue_length, stream_key)

            # 监控失败不影响 scheduler 主流程，只记录错误。
            except Exception as e:
                logger.error("Error in metrics monitor loop: %s", e, exc_info=True)

            # 每 15 秒采样一次队列长度。
            time.sleep(15)

    # 启动 scheduler 的消费者和后台监控。
    # 这是对外的整体启动入口。
    def start(self) -> None:
        # 如果开启并行 dispatch，启动时记录线程池大小。
        if self.enable_parallel_dispatch:
            logger.info(
                "Initializing dispatcher thread pool with %s workers",
                self.thread_pool_max_workers,
            )

        # 启动消息消费者线程或进程。
        self.start_consumer()

        # 启动后台 metrics 监控线程。
        self.start_background_monitor()

    # 启动后台队列长度监控线程。
    def start_background_monitor(self):
        # 如果监控线程已经存在且仍然存活，则不重复启动。
        if self._monitor_thread and self._monitor_thread.is_alive():
            return

        # 创建 ContextThread 运行 _monitor_loop。
        # daemon=True 表示主进程退出时该线程不会阻止进程结束。
        self._monitor_thread = ContextThread(
            target=self._monitor_loop, daemon=True, name="SchedulerMetricsMonitor"
        )

        # 启动监控线程。
        self._monitor_thread.start()

        # 记录启动成功。
        logger.info("Scheduler metrics monitor thread started.")

    # 启动消息消费者。
    # 根据 scheduler_startup_mode 决定使用进程还是线程。
    def start_consumer(self) -> None:
        # 如果已经运行，则不重复启动。
        if self._running:
            logger.warning("Memory Scheduler consumer is already running")
            return

        # 标记 scheduler 运行中。
        # _message_consumer 和 _monitor_loop 都依赖该标记控制循环。
        self._running = True

        # 如果配置为进程模式，则创建独立消费者进程。
        if self.scheduler_startup_mode == STARTUP_BY_PROCESS:
            self._consumer_process = multiprocessing.Process(
                target=self._message_consumer,
                daemon=True,
                name="MessageConsumerProcess",
            )

            # 启动消费者进程。
            self._consumer_process.start()

            # 记录进程启动。
            logger.info("Message consumer process started")

        # 否则使用线程模式启动消费者。
        else:
            self._consumer_thread = ContextThread(
                target=self._message_consumer,
                daemon=True,
                name="MessageConsumerThread",
            )

            # 启动消费者线程。
            self._consumer_thread.start()

            # 记录线程启动。
            logger.info("Message consumer thread started")

    # 停止消息消费者。
    # 它只负责消费者线程/进程，不负责 dispatcher 和监控器的完整关闭。
    def stop_consumer(self) -> None:
        # 如果当前没有运行，则无需停止。
        if not self._running:
            logger.warning("Memory Scheduler consumer is not running")
            return

        # 将运行标记置为 False，使消费者循环自然退出。
        self._running = False

        # 进程模式下，需要等待子进程退出，必要时强制 terminate。
        if self.scheduler_startup_mode == STARTUP_BY_PROCESS and self._consumer_process:
            # 如果消费者进程仍存活，先尝试优雅等待。
            if self._consumer_process.is_alive():
                self._consumer_process.join(timeout=5.0)

                # 如果 5 秒后仍未退出，则认为未能优雅停止。
                if self._consumer_process.is_alive():
                    logger.warning("Consumer process did not stop gracefully, terminating...")

                    # 强制终止子进程。
                    self._consumer_process.terminate()

                    # 再等待最多 2 秒。
                    self._consumer_process.join(timeout=2.0)

                    # 如果仍然存活，记录 error。
                    if self._consumer_process.is_alive():
                        logger.error("Consumer process could not be terminated")

                    # 否则记录已强制终止。
                    else:
                        logger.info("Consumer process terminated")

                # 如果 join 后进程已退出，记录正常停止。
                else:
                    logger.info("Consumer process stopped")

            # 清空进程引用。
            self._consumer_process = None

        # 线程模式下，等待消费者线程退出。
        elif self._consumer_thread and self._consumer_thread.is_alive():
            # 最多等待 5 秒。
            self._consumer_thread.join(timeout=5.0)

            # 如果仍存活，只能记录 warning，因为 Python 线程不能安全强杀。
            if self._consumer_thread.is_alive():
                logger.warning("Consumer thread did not stop gracefully")

            # 正常退出则记录信息。
            else:
                logger.info("Consumer thread stopped")

            # 清空线程引用。
            self._consumer_thread = None

        # 记录消费者停止完成。
        logger.info("Memory Scheduler consumer stopped")

    # 停止整个 scheduler。
    # 它会停止消费者、等待监控线程、关闭 dispatcher 和 dispatcher_monitor。
    def stop(self) -> None:
        # 如果 scheduler 当前未运行，则无需停止。
        if not self._running:
            logger.warning("Memory Scheduler is not running")
            return

        # 先停止消费者，避免继续拉取新消息。
        self.stop_consumer()

        # 如果监控线程存在，等待它短时间退出。
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)

        # 如果 dispatcher 存在，则关闭其线程池或内部资源。
        if self.dispatcher:
            logger.info("Shutting down dispatcher...")
            self.dispatcher.shutdown()

        # 如果 dispatcher_monitor 存在，则停止它。
        if self.dispatcher_monitor:
            logger.info("Shutting down monitor...")
            self.dispatcher_monitor.stop()

    # 暴露当前 dispatcher 注册的 handlers。
    # 这样外部可以查看当前有哪些 task label 对应的处理函数。
    @property
    def handlers(self) -> dict[str, Callable]:
        # dispatcher 未初始化时无法返回真实 handlers。
        if not self.dispatcher:
            logger.warning("Dispatcher is not initialized, returning empty handlers dict")
            return {}

        # 返回 dispatcher 内部 handlers 字典。
        return self.dispatcher.handlers

    # 注册一批 handler。
    # 支持两种形式：label -> callable，或 label -> (callable, priority, concurrency/worker 配置)。
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
        # dispatcher 不存在时无法注册 handler。
        if not self.dispatcher:
            logger.warning("Dispatcher is not initialized, cannot register handlers")
            return

        # 将注册动作委托给 dispatcher。
        self.dispatcher.register_handlers(handlers)

    # 注销一组 handler。
    # 返回每个 label 是否成功注销。
    def unregister_handlers(self, labels: list[str]) -> dict[str, bool]:
        # dispatcher 不存在时，所有 label 都视为注销失败。
        if not self.dispatcher:
            logger.warning("Dispatcher is not initialized, cannot unregister handlers")
            return dict.fromkeys(labels, False)

        # 委托 dispatcher 执行注销。
        return self.dispatcher.unregister_handlers(labels)

    # 获取当前正在运行的任务。
    # filter_func 可选，用于让 dispatcher 内部筛选任务。
    def get_running_tasks(self, filter_func: Callable | None = None) -> dict[str, dict]:
        # dispatcher 不存在时，没有可查询任务。
        if not self.dispatcher:
            logger.warning("Dispatcher is not initialized, returning empty tasks dict")
            return {}

        # 从 dispatcher 获取内部运行任务对象。
        running_tasks = self.dispatcher.get_running_tasks(filter_func=filter_func)

        # result 用于把内部 task_item 对象转换成普通 dict。
        # 这样对外返回更稳定，也更容易序列化。
        result = {}

        # 遍历每个运行中的任务。
        for task_id, task_item in running_tasks.items():
            # 抽取任务对象中的关键字段。
            result[task_id] = {
                "item_id": task_item.item_id,
                "user_id": task_item.user_id,
                "mem_cube_id": task_item.mem_cube_id,
                "task_info": task_item.task_info,
                "task_name": task_item.task_name,
                "start_time": task_item.start_time,
                "end_time": task_item.end_time,
                "status": task_item.status,
                "result": task_item.result,
                "error_message": task_item.error_message,
                "messages": task_item.messages,
            }

        # 返回普通 dict 形式的运行任务信息。
        return result

    # 获取任务状态监控器中的任务状态。
    # 这里直接委托 task_schedule_monitor。
    def get_tasks_status(self):
        return self.task_schedule_monitor.get_tasks_status()

    # 打印任务状态。
    # 如果传入 tasks_status，则打印传入内容；否则由 task_schedule_monitor 自行获取/打印。
    def print_tasks_status(self, tasks_status: dict | None = None) -> None:
        self.task_schedule_monitor.print_tasks_status(tasks_status=tasks_status)

    # 收集队列和 dispatcher 的运行统计信息。
    # 该方法主要用于健康检查、监控接口或调试输出。
    def _gather_queue_stats(self) -> dict:
        # 取出包装器内部真正的底层队列对象。
        # 注意 self.memos_message_queue 可能是 ScheduleTaskQueue 包装层，所以这里再取一层 memos_message_queue。
        memos_message_queue = self.memos_message_queue.memos_message_queue

        # stats 保存最终返回的统计字段。
        stats: dict[str, int | float | str] = {}

        # 记录当前是否使用 Redis 队列。
        stats["use_redis_queue"] = bool(self.use_redis_queue)

        # 本地队列才采集 qsize、unfinished_tasks、maxsize 和 utilization。
        # Redis 队列的统计方式通常不同，可能由其他路径提供。
        if not self.use_redis_queue:
            try:
                # 读取本地队列当前大小。
                stats["qsize"] = int(memos_message_queue.qsize())
            except Exception:
                # 读取失败时用 -1 表示未知。
                stats["qsize"] = -1

            try:
                # unfinished_tasks 是 Python queue.Queue 的常见字段。
                # 如果不存在或读取失败，则下面会兜底为 -1。
                stats["unfinished_tasks"] = int(
                    getattr(memos_message_queue, "unfinished_tasks", 0) or 0
                )
            except Exception:
                stats["unfinished_tasks"] = -1

            # 记录本地内部队列最大容量。
            stats["maxsize"] = int(self.max_internal_message_queue_size)

            try:
                # 计算队列利用率 qsize / maxsize。
                # maxsize 为 0 时用 1 兜底，避免除零。
                maxsize = int(self.max_internal_message_queue_size) or 1
                qsize = int(stats.get("qsize", 0))

                # 利用率限制在 [0.0, 1.0] 区间。
                stats["utilization"] = min(1.0, max(0.0, qsize / maxsize))
            except Exception:
                # 计算失败时给默认利用率 0。
                stats["utilization"] = 0.0

        try:
            # 从 dispatcher 获取运行统计。
            d_stats = self.dispatcher.stats()

            # 把 dispatcher 统计合并到 stats。
            stats.update(
                {
                    "running": int(d_stats.get("running", 0)),
                    "inflight": int(d_stats.get("inflight", 0)),
                    "handlers": int(d_stats.get("handlers", 0)),
                }
            )
        except Exception:
            # dispatcher 不可用或统计失败时，使用安全默认值。
            stats.update({"running": 0, "inflight": 0, "handlers": 0})

        # 返回队列和 dispatcher 的汇总统计。
        return stats
