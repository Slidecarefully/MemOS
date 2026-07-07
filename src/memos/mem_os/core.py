# 导入 JSON 模块，用于把记忆 ID、消息列表等结构化数据序列化后交给调度器。
import json
# 导入操作系统路径工具，用于判断传入的 MemCube 名称是否是本地路径。
import os
# 导入时间模块，用于记录检索、添加等关键流程的耗时日志。
import time

# 导入时间类型：datetime 用于时间戳，timezone 用于生成 UTC 创建时间。
from datetime import datetime, timezone
# 导入 Path，方便递归扫描文档目录并处理文件后缀。
from pathlib import Path
# 导入线程锁，保护调度器 setter 中的并发赋值。
from threading import Lock
# 导入类型标注工具，提升接口参数和返回值的可读性。
from typing import Any, Literal

# 导入 MOS 总配置类型，整个核心类依赖它读取用户、模型和记忆开关。
from memos.configs.mem_os import MOSConfig
# 导入带上下文传播能力的线程池，用于并行检索或并行写入记忆。
from memos.context.context import ContextThreadPoolExecutor
# 导入 LLM 工厂，根据配置动态创建聊天模型实例。
from memos.llms.factory import LLMFactory
# 导入日志构造函数，便于在核心流程中输出诊断信息。
from memos.log import get_logger
# 导入通用 MemCube 类型，它承载文本、激活、参数和偏好等记忆模块。
from memos.mem_cube.general import GeneralMemCube
# 导入记忆读取器工厂，用于从聊天或文档中抽取可写入的记忆。
from memos.mem_reader.factory import MemReaderFactory
# 导入通用调度器类型，用于类型校验和调度器属性声明。
from memos.mem_scheduler.general_scheduler import GeneralScheduler
# 导入调度器工厂，根据配置创建具体调度器实例。
from memos.mem_scheduler.scheduler_factory import SchedulerFactory
# 导入调度消息模型，所有提交给调度器的事件都会包装成它。
from memos.mem_scheduler.schemas.message_schemas import ScheduleMessageItem
# 导入调度任务标签常量，后续用标签区分查询、回答、添加、偏好等任务类型。
from memos.mem_scheduler.schemas.task_schemas import (
    # 表示同步添加文本记忆后的调度任务标签。
    ADD_TASK_LABEL,
    # 表示助手回答生成后的调度任务标签。
    ANSWER_TASK_LABEL,
    # 表示异步模式下让调度器继续读取/整理记忆的任务标签。
    MEM_READ_TASK_LABEL,
    # 表示异步添加偏好记忆的任务标签。
    PREF_ADD_TASK_LABEL,
    # 表示用户查询进入系统后的调度任务标签。
    QUERY_TASK_LABEL,
)
# 导入用户管理器和角色枚举，用来管理用户、权限和 MemCube 归属。
from memos.mem_user.user_manager import UserManager, UserRole
# 导入激活记忆条目类型，用在 get 的联合返回类型中。
from memos.memories.activation.item import ActivationMemoryItem
# 导入参数记忆条目类型，用在 get 的联合返回类型中。
from memos.memories.parametric.item import ParametricMemoryItem
# 导入文本记忆条目和元数据类型，写入文本记忆时会构造它们。
from memos.memories.textual.item import TextualMemoryItem, TextualMemoryMetadata
# 导入线程安全字典，多用户服务场景下保护 MemCube 容器。
from memos.memos_tools.thread_safe_dict_segment import OptimizedThreadSafeDict
# 导入查询改写提示词模板，用于结合历史对话重写用户问题。
from memos.templates.mos_prompts import QUERY_REWRITING_PROMPT
# 导入核心数据类型：聊天历史、消息列表和搜索结果结构。
from memos.types import ChatHistory, MessageList, MOSSearchResult


# 以当前模块名创建日志器，后续所有日志都通过它输出。
logger = get_logger(__name__)


# 定义 MOS 的核心调度层，它负责协调用户、MemCube、记忆检索、记忆写入和 LLM 对话。
class MOSCore:
    """
    The MOSCore (Memory Operating System Core) class manages multiple MemCube objects and their operations.
    It provides methods for creating, searching, updating, and deleting MemCubes, supporting multi-user scenarios.
    MOSCore acts as an operating system layer for handling and orchestrating MemCube instances.
    """

    # 初始化 MOSCore：串起配置、模型、记忆读取器、用户管理器、记忆立方体容器以及可选调度器。
    def __init__(self, config: MOSConfig, user_manager: UserManager | None = None):
        # 保存 MOS 总配置，后续模型、用户、会话和记忆功能开关都从这里读取。
        self.config = config
        # 记录配置中指定的默认用户 ID。
        self.user_id = config.user_id
        # 记录配置中指定的默认会话 ID，用于聊天历史和记忆元数据绑定。
        self.session_id = config.session_id
        # 根据聊天模型配置创建 LLM 实例，后续 chat 生成回复会调用它。
        self.chat_llm = LLMFactory.from_config(config.chat_model)
        # 根据记忆读取器配置创建抽取器，用来把聊天或文档转换成记忆。
        self.mem_reader = MemReaderFactory.from_config(config.mem_reader)
        # 初始化用户到聊天历史的映射，支持不同用户维护独立上下文。
        self.chat_history_manager: dict[str, ChatHistory] = {}
        # use thread safe dict for multi-user product-server scenario
        # 初始化 MemCube 容器；服务端多用户场景使用线程安全字典，单用户场景使用普通字典。
        self.mem_cubes: OptimizedThreadSafeDict[str, GeneralMemCube] = (
            # 如果外部传入 user_manager，说明更可能是多用户服务场景，因此启用线程安全容器。
            OptimizedThreadSafeDict() if user_manager is not None else {}
        )
        # 先为默认用户注册一份空聊天历史，保证 chat 时有上下文容器可用。
        self._register_chat_history()

        # Use provided user_manager or create a new one
        # 优先复用外部传入的用户管理器，让多个 MOSCore 可以共享用户/权限状态。
        if user_manager is not None:
            # 保存外部提供的用户管理器实例。
            self.user_manager = user_manager
        # 进入备选路径，通常用于默认值或异常情况处理。
        else:
            # 没有外部用户管理器时，创建一个新的管理器；若配置没有用户 ID，则使用 root。
            self.user_manager = UserManager(user_id=self.user_id if self.user_id else "root")

        # Validate user exists
        # 初始化阶段立即验证默认用户是否合法，避免核心对象带着无效用户继续运行。
        if not self.user_manager.validate_user(self.user_id):
            # 遇到不可恢复的非法状态时抛出 ValueError，让调用方明确知道输入或配置不满足要求。
            raise ValueError(
                f"User '{self.user_id}' does not exist or is inactive. Please create user first."
            )

        # Initialize mem_scheduler
        # 创建调度器赋值锁，避免多线程同时设置调度器造成状态不一致。
        self._mem_scheduler_lock = Lock()
        # 从配置读取是否启用记忆调度器；没有配置时默认关闭。
        self.enable_mem_scheduler = self.config.get("enable_mem_scheduler", False)
        # 只有配置启用调度器时，才创建并连接调度器相关组件。
        if self.enable_mem_scheduler:
            # 调用初始化函数创建调度器，并完成模块注入与启动。
            self._mem_scheduler = self._initialize_mem_scheduler()
            # 每次访问时都同步最新 MemCube 容器，避免调度器拿到过期引用。
            self._mem_scheduler.mem_cubes = self.mem_cubes
            # 把记忆读取器交给调度器，使异步任务能继续抽取或整理记忆。
            self._mem_scheduler.mem_reader = self.mem_reader
        # 进入备选路径，通常用于默认值或异常情况处理。
        else:
            # 调度器未启用时明确保存为空，后续访问可以据此判断。
            self._mem_scheduler: GeneralScheduler = None

        # 记录初始化完成日志，便于确认当前 MOSCore 绑定的用户。
        logger.info(f"MOS initialized for user: {self.user_id}")

    # 把下面的方法暴露成只读属性式访问，调用方可直接使用 self.mem_scheduler。
    @property
    # 暴露 memory scheduler 的访问入口：需要时懒加载，并保证调度器总是拿到最新的 MemCube 容器。
    def mem_scheduler(self) -> GeneralScheduler:
        """Lazy-loaded property for memory scheduler."""
        # 访问调度器时，如果配置启用但实例还不存在，就按需创建。
        if self.enable_mem_scheduler and self._mem_scheduler is None:
            # 触发调度器初始化流程。
            self._initialize_mem_scheduler()
        # 每次访问时都同步最新 MemCube 容器，避免调度器拿到过期引用。
        self._mem_scheduler.mem_cubes = self.mem_cubes
        # 返回当前调度器实例；如果未启用或初始化失败，则可能是 None。
        return self._mem_scheduler

    # 为 mem_scheduler 属性定义 setter，从而在外部赋值时统一做校验和同步。
    @mem_scheduler.setter
    # 暴露 memory scheduler 的访问入口：需要时懒加载，并保证调度器总是拿到最新的 MemCube 容器。
    def mem_scheduler(self, value: GeneralScheduler | None) -> None:
        """Setter for memory scheduler with validation.

        Args:
            value: GeneralScheduler instance or None to disable
        Raises:
            TypeError: If value is neither GeneralScheduler nor None
        """
        # 加锁包裹 setter 的整个修改过程，保证并发场景下状态更新原子化。
        with self._mem_scheduler_lock:
            # 如果外部传入了非空对象，必须确认它确实是 GeneralScheduler。
            if value is not None and not isinstance(value, GeneralScheduler):
                # 类型不符合约定时立即抛错，避免后续调用不存在的方法。
                raise TypeError(f"Expected GeneralScheduler or None, got {type(value)}")

            # 通过校验后，将内部调度器引用替换为新值。
            self._mem_scheduler = value
            # 每次访问时都同步最新 MemCube 容器，避免调度器拿到过期引用。
            self._mem_scheduler.mem_cubes = self.mem_cubes

            # 根据是否设置了有效调度器，输出不同级别的状态日志。
            if value:
                # 记录手动设置调度器成功。
                logger.info("Memory scheduler manually set")
            # 进入备选路径，通常用于默认值或异常情况处理。
            else:
                # 记录调度器被清空，用 debug 避免正常关闭时刷屏。
                logger.debug("Memory scheduler cleared")

    # 根据配置创建并启动记忆调度器，同时把 LLM、数据库等依赖注入调度器模块。
    def _initialize_mem_scheduler(self) -> GeneralScheduler:
        """Initialize the memory scheduler on first access."""
        # 如果配置层面禁用了调度器，就不再尝试创建实例。
        if not self.config.enable_mem_scheduler:
            # 记录调度器被配置关闭。
            logger.debug("Memory scheduler is disabled in config")
            # 把调度器状态重置为空，表示当前不可用。
            self._mem_scheduler = None
            # 返回当前调度器实例；如果未启用或初始化失败，则可能是 None。
            return self._mem_scheduler
        # 即便开关打开，也必须确认配置对象中有具体调度器配置。
        elif not hasattr(self.config, "mem_scheduler"):
            # 缺少调度器配置时写错误日志，提示配置不完整。
            logger.error("Config of Memory scheduler is not available")
            # 把调度器状态重置为空，表示当前不可用。
            self._mem_scheduler = None
            # 返回当前调度器实例；如果未启用或初始化失败，则可能是 None。
            return self._mem_scheduler
        # 进入备选路径，通常用于默认值或异常情况处理。
        else:
            # 记录调度器开始初始化，便于排查启动过程。
            logger.info("Initializing memory scheduler...")
            # 取出调度器专用配置，交给工厂创建具体实例。
            scheduler_config = self.config.mem_scheduler
            # 通过工厂方法构造调度器，屏蔽不同调度器实现的差异。
            self._mem_scheduler = SchedulerFactory.from_config(scheduler_config)
            # Validate required components
            # 调度器后续需要 LLM 能力，因此先检查 mem_reader 是否具备对应属性。
            if not hasattr(self.mem_reader, "llm"):
                # 抛出异常，向调用方明确报告当前操作无法继续。
                raise AttributeError(
                    f"Memory reader of type {type(self.mem_reader).__name__} "
                    "missing required 'llm' attribute"
                )
            # 进入备选路径，通常用于默认值或异常情况处理。
            else:
                # Configure scheduler general_modules
                # 把核心依赖注入调度器，让调度器能执行 LLM 处理和数据库操作。
                self._mem_scheduler.initialize_modules(
                    # 传入聊天 LLM，供调度任务需要生成或理解自然语言时使用。
                    chat_llm=self.chat_llm,
                    # 传入处理 LLM，通常用于记忆抽取、整理等后台处理。
                    process_llm=self.mem_reader.general_llm,
                    # 传入用户管理器持有的数据库引擎，使调度器能访问持久层。
                    db_engine=self.user_manager.engine,
                )
            # 调用调度器的启动方法。
            self._mem_scheduler.start()
            # 返回当前调度器实例；如果未启用或初始化失败，则可能是 None。
            return self._mem_scheduler

    # 尝试启动记忆调度器服务，返回布尔值表示启动是否成功。
    def mem_scheduler_on(self) -> bool:
        # 根据当前状态做分支判断，只有条件满足才进入后续业务逻辑。
        if not self.config.enable_mem_scheduler or self._mem_scheduler is None:
            # 调度器未配置或不存在时记录无法启动的原因。
            logger.error("Cannot start scheduler: disabled in configuration")

        # 进入可能失败的外部组件调用流程，用异常捕获保证接口返回布尔结果。
        try:
            # 调用调度器的启动方法。
            self._mem_scheduler.start()
            # 记录调度器服务启动成功。
            logger.info("Memory scheduler service started")
            # 操作成功时返回 True。
            return True
        # 捕获启动、停止或外部组件调用中的异常，避免异常直接向上泄漏。
        except Exception as e:
            # 记录启动失败的具体异常信息。
            logger.error(f"Failed to start scheduler: {e!s}")
            # 操作失败或条件不满足时返回 False。
            return False

    # 尝试关闭记忆调度器服务，返回布尔值表示关闭是否成功。
    def mem_scheduler_off(self) -> bool:
        # 如果配置层面禁用了调度器，就不再尝试创建实例。
        if not self.config.enable_mem_scheduler:
            # 调度器配置关闭时记录无法停止的原因。
            logger.error("Cannot stop scheduler: disabled in configuration")

        # 如果运行时根本没有调度器实例，就没有可停止的服务。
        if self._mem_scheduler is None:
            # 记录没有调度器实例可关闭。
            logger.warning("No scheduler instance to stop")
            # 操作失败或条件不满足时返回 False。
            return False

        # 进入可能失败的外部组件调用流程，用异常捕获保证接口返回布尔结果。
        try:
            # 调用调度器停止方法，释放后台调度资源。
            self._mem_scheduler.stop()
            # 记录调度器服务已经停止。
            logger.info("Memory scheduler service stopped")
            # 操作成功时返回 True。
            return True
        # 捕获启动、停止或外部组件调用中的异常，避免异常直接向上泄漏。
        except Exception as e:
            # 记录停止调度器失败的异常详情。
            logger.error(f"Failed to stop scheduler: {e!s}")
            # 操作失败或条件不满足时返回 False。
            return False

    # 预留的记忆重组器开启接口，目前还没有实现实际逻辑。
    def mem_reorganizer_on(self) -> bool:
        # 这里是占位实现，表示接口已经预留但当前版本还没有具体逻辑。
        pass

    # 关闭所有已加载 MemCube 中正在运行的文本记忆重组器，并等待其退出。
    def mem_reorganizer_off(self) -> bool:
        """temporally implement"""
        # 遍历当前已加载的所有 MemCube，对每个记忆库执行相同处理。
        for mem_cube in self.mem_cubes.values():
            # 输出日志，帮助观察这一步的运行状态或诊断问题。
            logger.info(f"try to close reorganizer for {mem_cube.text_mem.config.cube_id}")
            # 只有存在文本记忆模块且正在重组时，才需要关闭或等待重组器。
            if mem_cube.text_mem and mem_cube.text_mem.is_reorganize:
                # 输出日志，帮助观察这一步的运行状态或诊断问题。
                logger.info(f"close reorganizer for {mem_cube.text_mem.config.cube_id}")
                # 通知文本记忆管理器关闭重组器，准备结束后台任务。
                mem_cube.text_mem.memory_manager.close()
                # 阻塞等待重组器完成退出，确保状态收敛。
                mem_cube.text_mem.memory_manager.wait_reorganizer()

    # 等待所有正在重组的文本记忆管理器完成，用于同步重组流程。
    def mem_reorganizer_wait(self) -> bool:
        # 遍历当前已加载的所有 MemCube，对每个记忆库执行相同处理。
        for mem_cube in self.mem_cubes.values():
            # 输出日志，帮助观察这一步的运行状态或诊断问题。
            logger.info(f"try to close reorganizer for {mem_cube.text_mem.config.cube_id}")
            # 只有存在文本记忆模块且正在重组时，才需要关闭或等待重组器。
            if mem_cube.text_mem and mem_cube.text_mem.is_reorganize:
                # 输出日志，帮助观察这一步的运行状态或诊断问题。
                logger.info(f"close reorganizer for {mem_cube.text_mem.config.cube_id}")
                # 阻塞等待重组器完成退出，确保状态收敛。
                mem_cube.text_mem.memory_manager.wait_reorganizer()

    # 为指定用户/会话初始化一份聊天历史，后续检索和生成都会引用它。
    def _register_chat_history(
        # 计算并保存 self, user_id: str | None，供后续逻辑继续使用。
        self, user_id: str | None = None, session_id: str | None = None
    ) -> None:
        """Initialize chat history with user ID."""
        # 以 user_id 为键注册聊天历史；这里的 key 保留传入值，默认初始化时可能是 None。
        self.chat_history_manager[user_id] = ChatHistory(
            # ChatHistory 内部实际用户 ID 使用传入值；未传时回退到默认用户。
            user_id=user_id if user_id is not None else self.user_id,
            # ChatHistory 内部会话 ID 使用传入值；未传时回退到默认会话。
            session_id=session_id if session_id is not None else self.session_id,
            # 用 UTC 时间记录这份聊天历史的创建时刻。
            created_at=datetime.now(timezone.utc),
            # 新建历史时消息总数从 0 开始。
            total_messages=0,
            # 初始化空消息列表，后续 chat 会追加 user/assistant 消息。
            chat_history=[],
        )

    # 统一检查用户是否存在且处于可用状态，避免后续操作落到非法用户上。
    def _validate_user_exists(self, user_id: str) -> None:
        """Validate user exists and is active.

        Args:
            user_id (str): The user ID to validate.

        Raises:
            ValueError: If user doesn't exist or is inactive.
        """
        # 通过用户管理器确认目标用户存在且没有被禁用。
        if not self.user_manager.validate_user(user_id):
            # 遇到不可恢复的非法状态时抛出 ValueError，让调用方明确知道输入或配置不满足要求。
            raise ValueError(
                f"User '{user_id}' does not exist or is inactive. Please register the user first."
            )

    # 先验证用户，再验证用户是否有目标 MemCube 的访问权限。
    def _validate_cube_access(self, user_id: str, cube_id: str) -> None:
        """Validate user has access to the cube.

        Args:
            user_id (str): The user ID to validate.
            cube_id (str): The cube ID to validate.

        Raises:
            ValueError: If user doesn't have access to the cube.
        """
        # First validate user exists
        # 先验证用户本身存在，避免把权限问题和用户不存在混在一起。
        self._validate_user_exists(user_id)

        # Then validate cube access
        # 确认用户是否已经被授权访问目标 MemCube。
        if not self.user_manager.validate_user_cube_access(user_id, cube_id):
            # 遇到不可恢复的非法状态时抛出 ValueError，让调用方明确知道输入或配置不满足要求。
            raise ValueError(
                f"User '{user_id}' does not have access to cube '{cube_id}'. Please register the cube first or request access."
            )

    # 递归收集目录下支持的文档文件路径，供文档记忆导入流程使用。
    def _get_all_documents(self, path: str) -> list[str]:
        """Get all documents from path.

        Args:
            path (str): The path to get documents.

        Returns:
            list[str]: The list of documents.
        """
        # 准备收集符合后缀要求的文档路径。
        documents = []

        # 把字符串路径转换为 Path 对象，方便递归遍历。
        path_obj = Path(path)
        # 定义当前文档导入流程支持的文件类型白名单。
        doc_extensions = {".txt", ".pdf", ".json", ".md", ".ppt", ".pptx"}
        # 递归遍历目录下的所有文件和子目录。
        for file_path in path_obj.rglob("*"):
            # 只保留普通文件，并且文件后缀必须在支持列表中。
            if file_path.is_file() and (file_path.suffix.lower() in doc_extensions):
                # 把符合条件的文件路径转成字符串后加入结果列表。
                documents.append(str(file_path))
        # 返回收集到的全部文档路径。
        return documents

    # 完整聊天入口：检索记忆、构造系统提示、调用 LLM、更新历史，并把查询/回答提交给调度器。
    def chat(self, query: str, user_id: str | None = None, base_prompt: str | None = None) -> str:
        """
        Chat with the MOS.

        Args:
            query (str): The user's query.
            user_id (str, optional): The user ID for the chat session. Defaults to the user ID from the config.
            base_prompt (str, optional): A custom base prompt to use for the chat.
                It can be a template string with a `{memories}` placeholder.
                If not provided, a default prompt is used.

        Returns:
            str: The response from the MOS.
        """
        # 确定本次操作的目标用户：显式传入优先，否则使用 MOSCore 默认用户。
        target_user_id = user_id if user_id is not None else self.user_id
        # 读取目标用户可访问的 cube，用于未指定 cube_id 时选择默认目标。
        accessible_cubes = self.user_manager.get_user_cubes(target_user_id)
        # 提取可访问 MemCube 的 ID，后续过滤运行时容器时使用。
        user_cube_ids = [cube.cube_id for cube in accessible_cubes]
        # 如果目标用户还没有聊天历史，就先创建一份，避免后续访问失败。
        if target_user_id not in self.chat_history_manager:
            # 为目标用户初始化聊天历史。
            self._register_chat_history(target_user_id)

        # 取出目标用户历史对话，作为查询改写的上下文来源。
        chat_history = self.chat_history_manager[target_user_id]

        # 只有文本记忆功能开启且当前加载了 MemCube，才进行记忆检索。
        if self.config.enable_textual_memory and self.mem_cubes:
            # 准备汇总所有可访问 MemCube 检索出来的文本记忆。
            memories_all = []
            # 遍历已加载 MemCube，寻找当前用户可访问且带激活记忆的记忆库。
            for mem_cube_id, mem_cube in self.mem_cubes.items():
                # 跳过当前用户无权访问的 MemCube，保证多用户隔离。
                if mem_cube_id not in user_cube_ids:
                    # 当前对象不满足处理条件，直接跳到下一轮循环。
                    continue
                # 如果这个 MemCube 没有文本记忆模块，就无法参与文本检索。
                if not mem_cube.text_mem:
                    # 当前对象不满足处理条件，直接跳到下一轮循环。
                    continue

                # submit message to scheduler
                # 调度器启用且实例可用时，才向后台提交查询/回答/添加等事件。
                if self.enable_mem_scheduler and self.mem_scheduler is not None:
                    # 构造一条调度消息，把当前事件及其上下文交给后台调度器。
                    message_item = ScheduleMessageItem(
                        # 调度消息绑定目标用户，便于后台按用户隔离处理。
                        user_id=target_user_id,
                        # 调度消息绑定当前 MemCube，后台任务据此定位记忆库。
                        mem_cube_id=mem_cube_id,
                        # 把这条调度消息标记为用户查询事件。
                        label=QUERY_TASK_LABEL,
                        # 把原始用户查询作为调度消息内容。
                        content=query,
                        # 记录事件进入调度器的 UTC 时间戳。
                        timestamp=datetime.utcnow(),
                    )
                    # 把调度消息提交给调度器，统一使用列表形式以兼容批量接口。
                    self.mem_scheduler.submit_messages(messages=[message_item])

                # 在当前 MemCube 的文本记忆中执行语义/图搜索。
                memories = mem_cube.text_mem.search(
                    # 把用户原始查询传入记忆搜索。
                    query,
                    # 使用配置中的 top_k 控制返回记忆数量。
                    top_k=self.config.top_k,
                    # 传入搜索上下文信息，帮助底层记忆模块做个性化或过滤。
                    info={
                        # 上下文中带上用户 ID，便于记忆模块按用户过滤。
                        "user_id": target_user_id,
                        # 上下文中带上当前默认会话 ID。
                        "session_id": self.session_id,
                        # 把历史对话传给记忆模块，增强检索与抽取的上下文感知。
                        "chat_history": chat_history.chat_history,
                    },
                )
                # 把当前 MemCube 的检索结果追加到全局记忆列表中。
                memories_all.extend(memories)
            # 输出日志，帮助观察这一步的运行状态或诊断问题。
            logger.info(f"🧠 [Memory] Searched memories:\n{self._str_memories(memories_all)}\n")
            # 用检索到的记忆构造系统提示词，使 LLM 回答时能参考记忆上下文。
            system_prompt = self._build_system_prompt(memories_all, base_prompt=base_prompt)
        # 进入备选路径，通常用于默认值或异常情况处理。
        else:
            # 没有可用文本记忆时，只使用基础提示词构造系统提示。
            system_prompt = self._build_system_prompt(base_prompt=base_prompt)
        # 组装发送给聊天模型的完整消息列表：系统提示、历史对话和当前用户输入。
        current_messages = [
            # 第一条消息是系统提示，定义助手行为并承载记忆上下文。
            {"role": "system", "content": system_prompt},
            # 展开历史对话，保证模型看到当前会话上下文。
            *chat_history.chat_history,
            # 把当前用户问题追加到消息列表末尾。
            {"role": "user", "content": query},
        ]
        # 默认不使用激活记忆 KV 缓存，只有后续条件满足才会填充。
        past_key_values = None

        # 如果配置开启激活记忆，则尝试把 KV cache 作为模型生成的额外上下文。
        if self.config.enable_activation_memory:
            # 激活记忆只适用于 HuggingFace 后端，因此先检查当前聊天模型类型。
            if self.config.chat_model.backend not in ["huggingface", "huggingface_singleton"]:
                # 记录错误日志，说明当前流程遇到配置或运行时问题。
                logger.error(
                    "Activation memory only used for huggingface backend. Skipping activation memory."
                )
            # 进入备选路径，通常用于默认值或异常情况处理。
            else:
                # TODO this only one cubes
                # 遍历已加载 MemCube，寻找当前用户可访问且带激活记忆的记忆库。
                for mem_cube_id, mem_cube in self.mem_cubes.items():
                    # 跳过当前用户无权访问的 MemCube，保证多用户隔离。
                    if mem_cube_id not in user_cube_ids:
                        # 当前对象不满足处理条件，直接跳到下一轮循环。
                        continue
                    # 如果当前 MemCube 有激活记忆模块，就尝试读取其中的缓存。
                    if mem_cube.act_mem:
                        # 取出第一条激活记忆作为 KV cache；没有数据时返回 None。
                        kv_cache = next(iter(mem_cube.act_mem.get_all()), None)
                        # 从激活记忆对象中安全提取底层 memory 字段，作为模型 past_key_values。
                        past_key_values = (
                            # 只有 KV cache 存在且含 memory 属性时才使用，否则保持为空。
                            kv_cache.memory if (kv_cache and hasattr(kv_cache, "memory")) else None
                        )
                    # 当前实现只使用第一个符合条件的 MemCube，因此找到后就结束循环。
                    break
            # Generate response
            # 调用聊天模型生成回复，并把激活记忆缓存传入模型。
            response = self.chat_llm.generate(current_messages, past_key_values=past_key_values)
        # 进入备选路径，通常用于默认值或异常情况处理。
        else:
            # 不使用激活记忆时，直接基于消息列表生成回复。
            response = self.chat_llm.generate(current_messages)
        # 输出日志，帮助观察这一步的运行状态或诊断问题。
        logger.info(f"🤖 [Assistant] {response}\n")
        # 把当前用户输入追加进聊天历史，供后续轮次使用。
        chat_history.chat_history.append({"role": "user", "content": query})
        # 把助手回复追加进聊天历史，保持对话成对记录。
        chat_history.chat_history.append({"role": "assistant", "content": response})
        # 把更新后的聊天历史写回管理器；这里使用原始 user_id 作为键，保持原代码行为。
        self.chat_history_manager[user_id] = chat_history

        # submit message to scheduler
        # 回答生成后，对用户可访问的每个 MemCube 都提交一条回答事件。
        for accessible_mem_cube in accessible_cubes:
            # 取出当前可访问 MemCube 的 ID。
            mem_cube_id = accessible_mem_cube.cube_id
            # 根据 ID 从运行时容器取出对应 MemCube 实例。
            mem_cube = self.mem_cubes[mem_cube_id]
            # 调度器启用且实例可用时，才向后台提交查询/回答/添加等事件。
            if self.enable_mem_scheduler and self.mem_scheduler is not None:
                # 构造一条调度消息，把当前事件及其上下文交给后台调度器。
                message_item = ScheduleMessageItem(
                    # 调度消息绑定目标用户，便于后台按用户隔离处理。
                    user_id=target_user_id,
                    # 调度消息绑定当前 MemCube，后台任务据此定位记忆库。
                    mem_cube_id=mem_cube_id,
                    # 把这条调度消息标记为助手回答事件。
                    label=ANSWER_TASK_LABEL,
                    # 把模型生成的回复作为调度消息内容。
                    content=response,
                    # 记录事件进入调度器的 UTC 时间戳。
                    timestamp=datetime.utcnow(),
                )
                # 把调度消息提交给调度器，统一使用列表形式以兼容批量接口。
                self.mem_scheduler.submit_messages(messages=[message_item])

        # 把最终回复返回给调用方。
        return response

    # 把可选记忆拼进系统提示词；有占位符则替换，无占位符则兼容式追加。
    def _build_system_prompt(
        # 当前 MOSCore 实例本身。
        self,
        # 参数 memories 参与 _build_system_prompt 的业务流程，调用方可通过它改变本次操作的上下文或目标。
        memories: list[TextualMemoryItem] | list[str] | None = None,
        # 参数 base_prompt 参与 _build_system_prompt 的业务流程，调用方可通过它改变本次操作的上下文或目标。
        base_prompt: str | None = None,
        # 保留扩展参数入口，当前实现没有直接使用。
        **kwargs,
    ) -> str:
        """Build system prompt with optional memories context."""
        # 如果调用方没有提供自定义基础提示词，就使用内置默认提示。
        if base_prompt is None:
            # 构造默认系统提示词，说明助手能力以及如何自然使用记忆。
            base_prompt = (
                "You are a knowledgeable and helpful AI assistant. "
                "You have access to conversation memories that help you provide more personalized responses. "
                "Use the memories to understand the user's context, preferences, and past interactions. "
                "If memories are provided, reference them naturally when relevant, but don't explicitly mention having memories."
            )

        # 先准备空的记忆上下文字符串，后续有记忆时再填充。
        memory_context = ""
        # 只有传入了记忆列表，才需要把它们格式化进提示词。
        if memories:
            # 准备逐条存放格式化后的记忆文本。
            memory_list = []
            # 从 1 开始枚举记忆，方便在提示词中形成有序列表。
            for i, memory in enumerate(memories, 1):
                # 如果传入的是文本记忆对象，就取它的 memory 字段作为实际内容。
                if isinstance(memory, TextualMemoryItem):
                    # 提取文本记忆对象中的原始记忆文本。
                    text_memory = memory.memory
                # 进入备选路径，通常用于默认值或异常情况处理。
                else:
                    # 如果既不是 TextualMemoryItem 也不是字符串，就记录异常类型。
                    if not isinstance(memory, str):
                        # 记录意外记忆类型，帮助定位调用方传参问题。
                        logger.error("Unexpected memory type.")
                    # 非对象形式的记忆按字符串内容直接使用。
                    text_memory = memory
                # 把记忆格式化成编号列表项，便于拼接进提示词。
                memory_list.append(f"{i}. {text_memory}")
            # 用换行把所有记忆条目拼成一个上下文块。
            memory_context = "\n".join(memory_list)

        # 如果基础提示词显式提供 memories 占位符，就按模板位置插入记忆。
        if "{memories}" in base_prompt:
            # 把格式化后的记忆上下文替换进模板并返回。
            return base_prompt.format(memories=memory_context)
        # 没有占位符但有记忆时，走兼容逻辑，把记忆追加到提示词末尾。
        elif memories:
            # For backward compatibility, append memories if no placeholder is found
            # 给记忆上下文加标题，避免与原提示词正文混在一起。
            memory_context_with_header = "\n\n## Memories:\n" + memory_context
            # 返回基础提示词与记忆块拼接后的完整系统提示。
            return base_prompt + memory_context_with_header
        # 没有记忆或无需追加时，直接返回基础提示词。
        return base_prompt

    # 把记忆列表格式化成便于日志查看的字符串。
    def _str_memories(
        # 计算并保存 self, memories: list[TextualMemoryItem], mode: Literal["concise", "full"]，供后续逻辑继续使用。
        self, memories: list[TextualMemoryItem], mode: Literal["concise", "full"] = "full"
    ) -> str:
        """Format memories for display."""
        # 如果没有记忆结果，直接返回固定文本，避免格式化空列表。
        if not memories:
            # 用可读字符串表示没有命中的记忆。
            return "No memories."
        # 简洁模式只展示每条记忆的核心 memory 内容。
        if mode == "concise":
            # 把每条记忆内容按序号拼成多行字符串。
            return "\n".join(f"{i + 1}. {memory.memory}" for i, memory in enumerate(memories))
        # 完整模式展示整个记忆对象，便于调试元数据和结构。
        elif mode == "full":
            # 把每个记忆对象的完整表示按序号拼接输出。
            return "\n".join(f"{i + 1}. {memory}" for i, memory in enumerate(memories))

    # 清空某个用户的聊天历史，本质上是重新注册一份空历史。
    def clear_messages(self, user_id: str | None = None) -> None:
        """Clear chat history."""
        # 确定要清理历史的用户：显式传入优先，否则清理默认用户。
        user_id = user_id if user_id is not None else self.user_id
        self._register_chat_history(user_id)

    # 创建新用户；如果未传用户名，就用 user_id 作为默认用户名。
    def create_user(
        # 计算并保存 self, user_id: str, role: UserRole，供后续逻辑继续使用。
        self, user_id: str, role: UserRole = UserRole.USER, user_name: str | None = None
    ) -> str:
        """Create a new user.

        Args:
            user_name (str): Name of the user.
            role (UserRole): Role of the user.
            user_id (str, optional): Custom user ID.

        Returns:
            str: The created user ID.
        """
        # 如果没有提供用户显示名，就用 user_id 兜底。
        if not user_name:
            # 把 user_id 作为默认用户名，保证创建用户时名称不为空。
            user_name = user_id
        # 委托用户管理器创建用户，并返回新用户 ID。
        return self.user_manager.create_user(user_name, role, user_id)

    # 把用户管理器中的活跃用户对象整理成外部可读的字典列表。
    def list_users(self) -> list:
        """List all active users.

        Returns:
            list: List of user information dictionaries.
        """
        # 从用户管理器读取所有活跃用户对象。
        users = self.user_manager.list_users()
        # 将内部对象转换为列表结构返回给调用方。
        return [
            # 开始构造一个结构化字典或映射对象。
            {
                # 输出用户 ID。
                "user_id": user.user_id,
                # 输出用户名。
                "user_name": user.user_name,
                # 输出角色枚举值，而不是枚举对象本身。
                "role": user.role.value,
                # 把创建时间转换为 ISO 字符串，便于 JSON 序列化。
                "created_at": user.created_at.isoformat(),
                # 输出用户是否处于激活状态。
                "is_active": user.is_active,
            }
            # 遍历用户对象列表，逐个转换为字典。
            for user in users
        ]

    # 在用户管理器中创建一个新的 MemCube 记录，并返回 cube_id。
    def create_cube_for_user(
        # 当前 MOSCore 实例本身。
        self,
        # 参数 cube_name 参与 create_cube_for_user 的业务流程，调用方可通过它改变本次操作的上下文或目标。
        cube_name: str,
        # 参数 owner_id 参与 create_cube_for_user 的业务流程，调用方可通过它改变本次操作的上下文或目标。
        owner_id: str,
        # 参数 cube_path 参与 create_cube_for_user 的业务流程，调用方可通过它改变本次操作的上下文或目标。
        cube_path: str | None = None,
        # 参数 cube_id 参与 create_cube_for_user 的业务流程，调用方可通过它改变本次操作的上下文或目标。
        cube_id: str | None = None,
    ) -> str:
        """Create a new cube for the current user.

        Args:
            cube_name (str): Name of the cube.
            cube_path (str, optional): Path to the cube.
            cube_id (str, optional): Custom cube ID.

        Returns:
            str: The created cube ID.
        """
        # 委托用户管理器创建 MemCube 元数据记录。
        return self.user_manager.create_cube(cube_name, owner_id, cube_path, cube_id)

    # 把 MemCube 对象、本地路径或远程仓库注册进 MOS，并同步用户与数据库中的访问关系。
    def register_mem_cube(
        # 当前 MOSCore 实例本身。
        self,
        # 可传 MemCube 对象、本地路径或远程仓库名，注册逻辑会按类型分支处理。
        mem_cube_name_or_path: str | GeneralMemCube,
        # 可选 MemCube ID；为空时通常根据输入或用户权限推导。
        mem_cube_id: str | None = None,
        # 参数 user_id 参与 register_mem_cube 的业务流程，调用方可通过它改变本次操作的上下文或目标。
        user_id: str | None = None,
    ) -> None:
        """
        Register a MemCube with the MOS.

        Args:
            mem_cube_name_or_path (str): The name or path of the MemCube to register.
            mem_cube_id (str, optional): The identifier for the MemCube. If not provided, a default ID is used.
        """
        # 确定本次操作的目标用户：显式传入优先，否则使用 MOSCore 默认用户。
        target_user_id = user_id if user_id is not None else self.user_id
        # 搜索前确认目标用户有效，保证权限查询有意义。
        self._validate_user_exists(target_user_id)

        # 没有指定 MemCube 时，尝试从用户可访问列表中选择默认写入目标。
        if mem_cube_id is None:
            # 如果传入的是已构造好的 MemCube 对象，就按对象注册逻辑处理。
            if isinstance(mem_cube_name_or_path, GeneralMemCube):
                # 对象式注册没有路径名可用，因此用用户 ID 派生默认 cube_id。
                mem_cube_id = f"cube_{target_user_id}"
            # 进入备选路径，通常用于默认值或异常情况处理。
            else:
                # 路径或远程仓库名场景下，默认把输入字符串当作 cube_id。
                mem_cube_id = mem_cube_name_or_path

        # 只有已加载的 cube 才能参与本次运行时搜索。
        if mem_cube_id in self.mem_cubes:
            # 记录重复注册被跳过。
            logger.info(f"MemCube with ID {mem_cube_id} already in MOS, skip install.")
        # 进入备选路径，通常用于默认值或异常情况处理。
        else:
            # 如果传入的是已构造好的 MemCube 对象，就按对象注册逻辑处理。
            if isinstance(mem_cube_name_or_path, GeneralMemCube):
                # 把外部传入的 MemCube 对象直接挂到运行时容器。
                self.mem_cubes[mem_cube_id] = mem_cube_name_or_path
                # 记录为目标用户注册新 MemCube。
                logger.info(f"register new cube {mem_cube_id} for user {target_user_id}")
            # 如果输入字符串是本地存在的路径，就从本地目录加载 MemCube。
            elif os.path.exists(mem_cube_name_or_path):
                # 从本地目录初始化 MemCube 对象。
                mem_cube_obj = GeneralMemCube.init_from_dir(mem_cube_name_or_path)
                # 把新初始化的 MemCube 放入运行时容器。
                self.mem_cubes[mem_cube_id] = mem_cube_obj
            # 进入备选路径，通常用于默认值或异常情况处理。
            else:
                # 记录警告日志，表示系统进入非理想但可继续的路径。
                logger.warning(
                    f"MemCube {mem_cube_name_or_path} does not exist, try to init from remote repo."
                )
                # 本地路径不存在时，尝试把输入当作远程仓库并拉取初始化。
                mem_cube_obj = GeneralMemCube.init_from_remote_repo(mem_cube_name_or_path)
                # 把新初始化的 MemCube 放入运行时容器。
                self.mem_cubes[mem_cube_id] = mem_cube_obj
        # Check if cube already exists in database
        # 查询数据库/用户管理器中是否已经存在该 MemCube 的元数据记录。
        existing_cube = self.user_manager.get_cube(mem_cube_id)

        # check the embedder is it consistent with MOSConfig
        # 先检查对象是否具备某个属性，再访问它以避免 AttributeError。
        if hasattr(
            # 检查 MemCube 文本记忆配置中是否声明了 embedder。
            self.mem_cubes[mem_cube_id].text_mem.config, "embedder"
        # 如果 cube 自带 embedder 与 MOSConfig 中的 reader embedder 不一致，就需要提示。
        ) and self.config.mem_reader.config.embedder != (
            # 用海象运算符同时读取 cube 的 embedder 并保存给日志使用。
            cube_embedder := self.mem_cubes[mem_cube_id].text_mem.config.embedder
        ):
            # 记录警告日志，表示系统进入非理想但可继续的路径。
            logger.warning(
                f"Cube Embedder is not consistent with MOSConfig for cube: {mem_cube_id}, will use Cube Embedder: {cube_embedder}"
            )

        # 如果数据库里已有该 MemCube 记录，只需要处理用户访问关系。
        if existing_cube:
            # Cube exists, just add user to cube if not already associated
            # 如果目标用户还没有访问权限，就把用户加入该 MemCube。
            if not self.user_manager.validate_user_cube_access(target_user_id, mem_cube_id):
                # 请求用户管理器建立用户与 MemCube 的授权关系。
                success = self.user_manager.add_user_to_cube(target_user_id, mem_cube_id)
                # 根据授权操作返回值判断是否添加成功。
                if success:
                    # 记录用户成功获得已有 MemCube 权限。
                    logger.info(f"User {target_user_id} added to existing cube {mem_cube_id}")
                # 进入备选路径，通常用于默认值或异常情况处理。
                else:
                    # 记录用户授权失败。
                    logger.error(f"Failed to add user {target_user_id} to cube {mem_cube_id}")
            # 进入备选路径，通常用于默认值或异常情况处理。
            else:
                # 记录用户原本就拥有该 MemCube 权限。
                logger.info(f"User {target_user_id} already has access to cube {mem_cube_id}")
        # 进入备选路径，通常用于默认值或异常情况处理。
        else:
            # Cube doesn't exist, create it
            # 数据库中没有该 MemCube 记录时，创建新的 cube 元数据。
            self.create_cube_for_user(
                # 计算并保存 cube_name，供后续逻辑继续使用。
                cube_name=mem_cube_name_or_path
                # 根据当前状态做分支判断，只有条件满足才进入后续业务逻辑。
                if not isinstance(mem_cube_name_or_path, GeneralMemCube)
                else mem_cube_id,
                # 把目标用户设置为新 MemCube 的拥有者。
                owner_id=target_user_id,
                # 显式传入最终确定的 cube_id。
                cube_id=mem_cube_id,
                # 计算并保存 cube_path，供后续逻辑继续使用。
                cube_path=mem_cube_name_or_path
                # 根据当前状态做分支判断，只有条件满足才进入后续业务逻辑。
                if not isinstance(mem_cube_name_or_path, GeneralMemCube)
                else "init",
            )
            # 记录为目标用户注册新 MemCube。
            logger.info(f"register new cube {mem_cube_id} for user {target_user_id}")

    # 从当前 MOS 运行时容器中移除一个已加载的 MemCube。
    def unregister_mem_cube(self, mem_cube_id: str, user_id: str | None = None) -> None:
        """
        Unregister a MemCube by its identifier.

        Args:
            mem_cube_id (str): The identifier of the MemCube to unregister.
        """
        # 只有已加载的 cube 才能参与本次运行时搜索。
        if mem_cube_id in self.mem_cubes:
            # 从运行时容器删除 MemCube 引用；这只是注销加载状态，不等同于删除持久化数据。
            del self.mem_cubes[mem_cube_id]
        # 进入备选路径，通常用于默认值或异常情况处理。
        else:
            # 目标 MemCube 没有加载时，抛错提示调用方。
            raise ValueError(f"MemCube with ID {mem_cube_id} does not exist.")

    # 跨当前用户可访问的 MemCube 检索文本记忆和偏好记忆，并按类型汇总结果。
    def search(
        # 当前 MOSCore 实例本身。
        self,
        # 参数 query 参与 search 的业务流程，调用方可通过它改变本次操作的上下文或目标。
        query: str,
        # 参数 user_id 参与 search 的业务流程，调用方可通过它改变本次操作的上下文或目标。
        user_id: str | None = None,
        # 参数 install_cube_ids 参与 search 的业务流程，调用方可通过它改变本次操作的上下文或目标。
        install_cube_ids: list[str] | None = None,
        # 参数 top_k 参与 search 的业务流程，调用方可通过它改变本次操作的上下文或目标。
        top_k: int | None = None,
        # 参数 mode 参与 search 的业务流程，调用方可通过它改变本次操作的上下文或目标。
        mode: Literal["fast", "fine"] = "fast",
        # 参数 internet_search 参与 search 的业务流程，调用方可通过它改变本次操作的上下文或目标。
        internet_search: bool = False,
        # 参数 moscube 参与 search 的业务流程，调用方可通过它改变本次操作的上下文或目标。
        moscube: bool = False,
        # 参数 session_id 参与 search 的业务流程，调用方可通过它改变本次操作的上下文或目标。
        session_id: str | None = None,
        # 保留扩展参数入口，当前实现没有直接使用。
        **kwargs,
    ) -> MOSSearchResult:
        """
        Search for textual memories across all registered MemCubes.

        Args:
            query (str): The search query.
            user_id (str, optional): The identifier of the user to search for.
                If None, the default user is used.
            install_cube_ids (list[str], optional): The list of MemCube IDs to install.
                If None, all MemCube for the user is used.

        Returns:
            MemoryResult: A dictionary containing the search results.
        """
        # 确定本次检索绑定的会话 ID：显式传入优先，否则使用默认会话。
        target_session_id = session_id if session_id is not None else self.session_id
        # 确定本次操作的目标用户：显式传入优先，否则使用 MOSCore 默认用户。
        target_user_id = user_id if user_id is not None else self.user_id

        # 搜索前确认目标用户有效，保证权限查询有意义。
        self._validate_user_exists(target_user_id)
        # Get all cubes accessible by the target user
        # 读取目标用户可访问的 cube，用于未指定 cube_id 时选择默认目标。
        accessible_cubes = self.user_manager.get_user_cubes(target_user_id)
        # 提取可访问 MemCube 的 ID，后续过滤运行时容器时使用。
        user_cube_ids = [cube.cube_id for cube in accessible_cubes]

        # 记录流程状态或诊断信息，方便追踪运行路径和耗时。
        logger.info(
            f"User {target_user_id} has access to {len(user_cube_ids)} cubes: {user_cube_ids}"
        )
        # 如果目标用户还没有聊天历史，就先创建一份，避免后续访问失败。
        if target_user_id not in self.chat_history_manager:
            # 为目标用户初始化聊天历史。
            self._register_chat_history(target_user_id)
        # 取出目标用户历史对话，作为查询改写的上下文来源。
        chat_history = self.chat_history_manager[target_user_id]

        # Create search filter if session_id is provided
        # 默认不加搜索过滤条件，表示跨会话检索。
        search_filter = None
        # 如果调用方指定了会话 ID，则只搜索该会话相关记忆。
        if session_id is not None:
            # 构造会话过滤条件，交给底层记忆搜索实现。
            search_filter = {"session_id": session_id}

        # 初始化标准搜索结果结构，不同类型记忆分别放入不同列表。
        result: MOSSearchResult = {
            # 文本记忆搜索结果列表。
            "text_mem": [],
            # 激活记忆结果列表；当前 search 方法主要预留该字段。
            "act_mem": [],
            # 参数记忆结果列表；当前 search 方法主要预留该字段。
            "para_mem": [],
            # 偏好记忆搜索结果列表。
            "pref_mem": [],
        }
        # 如果调用方没有限定搜索的 cube，则默认搜索当前用户可访问的全部 cube。
        if install_cube_ids is None:
            # 把可访问 cube ID 作为本次检索范围。
            install_cube_ids = user_cube_ids
        # create exist dict in mem_cubes and avoid  one search slow
        # 构造一个只包含已加载目标 cube 的临时字典，减少后续重复查找。
        tmp_mem_cubes = {}
        # 记录筛选已加载 cube 的开始时间，用于耗时日志。
        time_start_cube_get = time.time()
        # 逐个检查本次检索范围内的 cube 是否已经加载到运行时。
        for mem_cube_id in install_cube_ids:
            # 只有已加载的 cube 才能参与本次运行时搜索。
            if mem_cube_id in self.mem_cubes:
                # 把已加载 cube 放进临时集合，后续直接遍历它。
                tmp_mem_cubes[mem_cube_id] = self.mem_cubes.get(mem_cube_id)
        # 记录流程状态或诊断信息，方便追踪运行路径和耗时。
        logger.info(
            f"time search: transform cube time user_id: {target_user_id} time is: {time.time() - time_start_cube_get}"
        )

        # 对每个可搜索的 MemCube 分别执行文本和偏好检索。
        for mem_cube_id, mem_cube in tmp_mem_cubes.items():
            # Define internal functions for parallel search execution
            # 内部并发任务：在单个 MemCube 的文本记忆中执行搜索。
            def search_textual_memory(cube_id, cube):
                # 开始组合多个前置条件，只有全部满足才进入核心逻辑。
                if (
                    # 确认该 cube 仍在本次请求允许的搜索范围内。
                    (cube_id in install_cube_ids)
                    # 确认该 cube 拥有文本记忆模块。
                    and (cube.text_mem is not None)
                    # 确认全局配置开启文本记忆写入。
                    and self.config.enable_textual_memory
                ):
                    # 记录当前流程开始时间，用于后续性能日志。
                    time_start = time.time()
                    # 计算并保存 memories，供后续逻辑继续使用。
                    memories = cube.text_mem.search(
                        # 把用户原始查询传入记忆搜索。
                        query,
                        # 优先使用调用方传入的 top_k；未传时回退到配置默认值。
                        top_k=top_k if top_k else self.config.top_k,
                        # 把检索模式传给底层搜索，fast/fine 会影响速度与精细度。
                        mode=mode,
                        # 根据 internet_search 参数决定是否显式关闭底层联网搜索。
                        manual_close_internet=not internet_search,
                        # 传入搜索上下文信息，帮助底层记忆模块做个性化或过滤。
                        info={
                            # 上下文中带上用户 ID，便于记忆模块按用户过滤。
                            "user_id": target_user_id,
                            # 上下文中带上本次解析后的会话 ID。
                            "session_id": target_session_id,
                            # 把历史对话传给记忆模块，增强检索与抽取的上下文感知。
                            "chat_history": chat_history.chat_history,
                        },
                        # 把 moscube 标记透传给底层搜索，决定是否启用特定搜索能力。
                        moscube=moscube,
                        # 把可选会话过滤条件透传给底层搜索。
                        search_filter=search_filter,
                    )
                    # 记录单个检索任务结束时间，用于计算耗时。
                    search_time_end = time.time()
                    # 记录流程状态或诊断信息，方便追踪运行路径和耗时。
                    logger.info(
                        f"🧠 [Memory] Searched memories from {cube_id}:\n{self._str_memories(memories)}\n"
                    )
                    # 记录流程状态或诊断信息，方便追踪运行路径和耗时。
                    logger.info(
                        f"time search graph: search graph time user_id: {target_user_id} time is: {search_time_end - time_start}"
                    )
                    # 把 cube_id 和搜索结果绑定返回，方便上层汇总时保留来源。
                    return {"cube_id": cube_id, "memories": memories}
                # 当前 cube 不满足检索条件时返回 None，汇总阶段会跳过。
                return None

            # 内部并发任务：在单个 MemCube 的偏好记忆中执行搜索。
            def search_preference_memory(cube_id, cube):
                # 开始组合多个前置条件，只有全部满足才进入核心逻辑。
                if (
                    # 确认该 cube 仍在本次请求允许的搜索范围内。
                    (cube_id in install_cube_ids)
                    # 确认该 cube 拥有偏好记忆模块。
                    and (cube.pref_mem is not None)
                    # 确认全局配置开启偏好记忆能力。
                    and self.config.enable_preference_memory
                ):
                    # 记录当前流程开始时间，用于后续性能日志。
                    time_start = time.time()
                    # 在当前 MemCube 的偏好记忆模块中执行搜索。
                    memories = cube.pref_mem.search(
                        # 把用户原始查询传入记忆搜索。
                        query,
                        # 优先使用调用方传入的 top_k；未传时回退到配置默认值。
                        top_k=top_k if top_k else self.config.top_k,
                        # 传入搜索上下文信息，帮助底层记忆模块做个性化或过滤。
                        info={
                            # 上下文中带上用户 ID，便于记忆模块按用户过滤。
                            "user_id": target_user_id,
                            # 上下文中带上当前默认会话 ID。
                            "session_id": self.session_id,
                            # 把历史对话传给记忆模块，增强检索与抽取的上下文感知。
                            "chat_history": chat_history.chat_history,
                        },
                    )
                    # 记录单个检索任务结束时间，用于计算耗时。
                    search_time_end = time.time()
                    # 记录流程状态或诊断信息，方便追踪运行路径和耗时。
                    logger.info(
                        f"🧠 [Memory] Searched preferences from {cube_id}:\n{self._str_memories(memories)}\n"
                    )
                    # 记录流程状态或诊断信息，方便追踪运行路径和耗时。
                    logger.info(
                        f"time search pref: search pref time user_id: {target_user_id} time is: {search_time_end - time_start}"
                    )
                    # 把 cube_id 和搜索结果绑定返回，方便上层汇总时保留来源。
                    return {"cube_id": cube_id, "memories": memories}
                # 当前 cube 不满足检索条件时返回 None，汇总阶段会跳过。
                return None

            # Execute both search functions in parallel
            # 用两个线程并行执行文本记忆和偏好记忆搜索，提高单个 cube 的检索效率。
            with ContextThreadPoolExecutor(max_workers=2) as executor:
                # 提交文本记忆搜索任务，并拿到 Future 以便等待结果。
                text_future = executor.submit(search_textual_memory, mem_cube_id, mem_cube)
                # 提交偏好记忆搜索任务，并拿到 Future 以便等待结果。
                pref_future = executor.submit(search_preference_memory, mem_cube_id, mem_cube)

                # Wait for both tasks to complete and collect results
                # 等待文本记忆搜索完成，并取回结果。
                text_result = text_future.result()
                # 等待偏好记忆搜索完成，并取回结果。
                pref_result = pref_future.result()

                # Add results to the main result dictionary
                # 只有子任务实际返回结果时，才追加到总结果中。
                if text_result is not None:
                    # 把文本记忆搜索结果加入 text_mem 列表。
                    result["text_mem"].append(text_result)
                # 只有偏好记忆搜索返回结果时，才追加到总结果中。
                if pref_result is not None:
                    # 把偏好记忆搜索结果加入 pref_mem 列表。
                    result["pref_mem"].append(pref_result)

        # 返回按记忆类型组织好的搜索结果。
        return result

    # 统一的记忆写入入口：支持会话消息、单条文本记忆和文档路径三种输入。
    def add(
        # 当前 MOSCore 实例本身。
        self,
        # 参数 messages 参与 add 的业务流程，调用方可通过它改变本次操作的上下文或目标。
        messages: MessageList | None = None,
        # 参数 memory_content 参与 add 的业务流程，调用方可通过它改变本次操作的上下文或目标。
        memory_content: str | None = None,
        # 参数 doc_path 参与 add 的业务流程，调用方可通过它改变本次操作的上下文或目标。
        doc_path: str | None = None,
        # 可选 MemCube ID；为空时通常根据输入或用户权限推导。
        mem_cube_id: str | None = None,
        # 参数 user_id 参与 add 的业务流程，调用方可通过它改变本次操作的上下文或目标。
        user_id: str | None = None,
        # 参数 session_id 参与 add 的业务流程，调用方可通过它改变本次操作的上下文或目标。
        session_id: str | None = None,
        # 可选任务 ID；原注释保留，用于异步调度链路追踪。
        task_id: str | None = None,  # New: Add task_id parameter
        # 保留扩展参数入口，当前实现没有直接使用。
        **kwargs,
    ) -> None:
        """
        Add textual memories to a MemCube.

        Args:
            messages (Union[MessageList, str]): The path to a document or a list of messages.
            memory_content (str, optional): The content of the memory to add.
            doc_path (str, optional): The path to the document associated with the memory.
            mem_cube_id (str, optional): The identifier of the MemCube to add the memories to.
                If None, the default MemCube for the user is used.
            user_id (str, optional): The identifier of the user to add the memories to.
                If None, the default user is used.
            session_id (str, optional): session_id
        """
        # user input messages
        # 写入记忆时至少要提供一种输入来源：消息、文本内容或文档路径。
        assert (messages is not None) or (memory_content is not None) or (doc_path is not None), (
            "messages_or_doc_path or memory_content or doc_path must be provided."
        )
        # TODO: asure that session_id is a valid string
        # 记录当前流程开始时间，用于后续性能日志。
        time_start = time.time()

        # 确定本次写入绑定的会话 ID：传入非空值优先，否则使用默认会话。
        target_session_id = session_id if session_id else self.session_id
        # 确定本次操作的目标用户：显式传入优先，否则使用 MOSCore 默认用户。
        target_user_id = user_id if user_id is not None else self.user_id
        # 没有指定 MemCube 时，尝试从用户可访问列表中选择默认写入目标。
        if mem_cube_id is None:
            # Try to find a default cube for the user
            # 读取目标用户可访问的 cube，用于未指定 cube_id 时选择默认目标。
            accessible_cubes = self.user_manager.get_user_cubes(target_user_id)
            # 如果用户没有任何可访问 cube，就无法确定默认写入位置。
            if not accessible_cubes:
                # 遇到不可恢复的非法状态时抛出 ValueError，让调用方明确知道输入或配置不满足要求。
                raise ValueError(
                    f"No accessible cubes found for user '{target_user_id}'. Please register a cube first."
                )
            # 当前实现临时使用第一个可访问 cube 作为默认目标，原 TODO 提醒未来应支持更合理选择。
            mem_cube_id = accessible_cubes[0].cube_id  # TODO not only first
        # 进入备选路径，通常用于默认值或异常情况处理。
        else:
            # 读取/更新/删除前确认用户拥有目标 MemCube 的访问权限。
            self._validate_cube_access(target_user_id, mem_cube_id)
        # 记录流程状态或诊断信息，方便追踪运行路径和耗时。
        logger.info(
            f"time add: get mem_cube_id time user_id: {target_user_id} time is: {time.time() - time_start}"
        )

        # 权限存在不代表运行时已加载；这里确认目标 cube 已在内存中。
        if mem_cube_id not in self.mem_cubes:
            # 目标 MemCube 未加载时抛错，提示调用方先注册或加载。
            raise ValueError(f"MemCube '{mem_cube_id}' is not loaded. Please register.")

        # 读取文本记忆模块的同步模式，后续决定立即处理还是提交给调度器。
        sync_mode = self.mem_cubes[mem_cube_id].text_mem.mode
        # 异步模式下，添加后的进一步处理交给 MEM_READ 调度任务。
        if sync_mode == "async":
            # 异步模式下必须有可用调度器，否则后台任务无法继续执行。
            assert self.mem_scheduler is not None, (
                "Mem-Scheduler must be working when use asynchronous memory adding."
            )
        # 记录当前记忆读取/写入模式。
        logger.debug(f"Mem-reader mode is: {sync_mode}")

        # 内部并发任务：把聊天消息转换并写入文本记忆。
        def process_textual_memory():
            # 开始组合多个前置条件，只有全部满足才进入核心逻辑。
            if (
                # 只有传入聊天消息时，才会执行消息到记忆的转换/写入。
                (messages is not None)
                # 确认全局配置开启文本记忆写入。
                and self.config.enable_textual_memory
                # 确认目标 MemCube 存在文本记忆模块。
                and self.mem_cubes[mem_cube_id].text_mem
            ):
                # 只有非 tree_text 后端支持直接更新文本记忆。
                if self.mem_cubes[mem_cube_id].config.text_mem.backend != "tree_text":
                    # 准备收集待写入的 TextualMemoryItem 列表。
                    add_memory = []
                    # 为即将写入的文本记忆构造统一元数据。
                    metadata = TextualMemoryMetadata(
                        # 元数据记录用户、会话和来源，方便后续检索过滤与溯源。
                        user_id=target_user_id, session_id=target_session_id, source="conversation"
                    )
                    # 逐条处理输入消息，把每条消息内容转换成一条文本记忆。
                    for message in messages:
                        # 把构造好的文本记忆对象追加到待写入列表。
                        add_memory.append(
                            # 使用消息 content 字段作为记忆正文，并绑定统一元数据。
                            TextualMemoryItem(memory=message["content"], metadata=metadata)
                        )
                    # 把整理好的文本记忆列表写入目标 MemCube 的文本记忆模块。
                    self.mem_cubes[mem_cube_id].text_mem.add(add_memory)
                # 进入备选路径，通常用于默认值或异常情况处理。
                else:
                    # tree_text 后端期望批量对话格式，因此把当前消息列表再包一层列表。
                    messages_list = [messages]
                    # 调用 mem_reader 从消息或文档中抽取结构化记忆。
                    memories = self.mem_reader.get_memory(
                        # 把待抽取的消息批次传入 mem_reader。
                        messages_list,
                        # 声明输入类型是聊天对话。
                        type="chat",
                        # 计算并保存 info，供后续逻辑继续使用。
                        info={"user_id": target_user_id, "session_id": target_session_id},
                        # 异步模式先用 fast 快速抽取；同步模式用 fine 更精细地抽取。
                        mode="fast" if sync_mode == "async" else "fine",
                    )
                    # mem_reader 返回批次嵌套列表，这里展平成单层记忆列表。
                    memories_flatten = [m for m_list in memories for m in m_list]
                    # 把抽取后的文本记忆写入 text_mem，并接收新生成的记忆 ID。
                    mem_ids: list[str] = self.mem_cubes[mem_cube_id].text_mem.add(memories_flatten)
                    # 记录流程状态或诊断信息，方便追踪运行路径和耗时。
                    logger.info(
                        f"Added memory user {target_user_id} to memcube {mem_cube_id}: {mem_ids}"
                    )
                    # submit messages for scheduler
                    # 调度器启用且实例可用时，才向后台提交查询/回答/添加等事件。
                    if self.enable_mem_scheduler and self.mem_scheduler is not None:
                        # 异步模式下，添加后的进一步处理交给 MEM_READ 调度任务。
                        if sync_mode == "async":
                            # 构造一条调度消息，把当前事件及其上下文交给后台调度器。
                            message_item = ScheduleMessageItem(
                                # 调度消息绑定目标用户，便于后台按用户隔离处理。
                                user_id=target_user_id,
                                # 调度消息绑定当前 MemCube，后台任务据此定位记忆库。
                                mem_cube_id=mem_cube_id,
                                # 把这条任务标记为异步记忆读取/整理任务。
                                label=MEM_READ_TASK_LABEL,
                                # 把新增记忆 ID 序列化成 JSON 字符串作为任务内容。
                                content=json.dumps(mem_ids),
                                # 记录事件进入调度器的 UTC 时间戳。
                                timestamp=datetime.utcnow(),
                                # 保留外部传入的任务 ID，便于异步任务链路追踪。
                                task_id=task_id,
                            )
                            # 把调度消息提交给调度器，统一使用列表形式以兼容批量接口。
                            self.mem_scheduler.submit_messages(messages=[message_item])
                        # 进入备选路径，通常用于默认值或异常情况处理。
                        else:
                            # 构造一条调度消息，把当前事件及其上下文交给后台调度器。
                            message_item = ScheduleMessageItem(
                                # 调度消息绑定目标用户，便于后台按用户隔离处理。
                                user_id=target_user_id,
                                # 调度消息绑定当前 MemCube，后台任务据此定位记忆库。
                                mem_cube_id=mem_cube_id,
                                # 把这条任务标记为同步添加记忆后的后处理任务。
                                label=ADD_TASK_LABEL,
                                # 把新增记忆 ID 序列化成 JSON 字符串作为任务内容。
                                content=json.dumps(mem_ids),
                                # 记录事件进入调度器的 UTC 时间戳。
                                timestamp=datetime.utcnow(),
                                # 保留外部传入的任务 ID，便于异步任务链路追踪。
                                task_id=task_id,
                            )
                            # 记录流程状态或诊断信息，方便追踪运行路径和耗时。
                            logger.info(
                                # 计算并保存 f"[DIAGNOSTIC] core.add: Submitting message to scheduler: {message_item.model_dump_json(indent，供后续逻辑继续使用。
                                f"[DIAGNOSTIC] core.add: Submitting message to scheduler: {message_item.model_dump_json(indent=2)}"
                            )
                            # 把调度消息提交给调度器，统一使用列表形式以兼容批量接口。
                            self.mem_scheduler.submit_messages(messages=[message_item])

        # 内部并发任务：从聊天消息中提取并写入偏好记忆。
        def process_preference_memory():
            # 开始组合多个前置条件，只有全部满足才进入核心逻辑。
            if (
                # 只有传入聊天消息时，才会执行消息到记忆的转换/写入。
                (messages is not None)
                # 确认全局配置开启偏好记忆能力。
                and self.config.enable_preference_memory
                # 确认目标 MemCube 存在偏好记忆模块。
                and self.mem_cubes[mem_cube_id].pref_mem
            ):
                # tree_text 后端期望批量对话格式，因此把当前消息列表再包一层列表。
                messages_list = [messages]
                # 同步模式下立即从聊天消息中抽取偏好并写入。
                if sync_mode == "sync":
                    # 调用偏好记忆模块从消息中抽取偏好候选。
                    pref_memories = self.mem_cubes[mem_cube_id].pref_mem.get_memory(
                        # 把待抽取的消息批次传入 mem_reader。
                        messages_list,
                        # 声明输入类型是聊天对话。
                        type="chat",
                        # 传入搜索上下文信息，帮助底层记忆模块做个性化或过滤。
                        info={
                            # 上下文中带上用户 ID，便于记忆模块按用户过滤。
                            "user_id": target_user_id,
                            # 上下文中带上当前默认会话 ID。
                            "session_id": self.session_id,
                            # 把目标 MemCube ID 放入上下文，方便偏好模块定位或记录来源。
                            "mem_cube_id": mem_cube_id,
                        },
                    )
                    # 把抽取出的偏好记忆写入 pref_mem，并获取新偏好 ID。
                    pref_ids = self.mem_cubes[mem_cube_id].pref_mem.add(pref_memories)
                    # 记录流程状态或诊断信息，方便追踪运行路径和耗时。
                    logger.info(
                        f"Added preferences user {target_user_id} to memcube {mem_cube_id}: {pref_ids}"
                    )
                # 异步模式下不立即抽取偏好，而是把原消息提交给调度器处理。
                elif sync_mode == "async":
                    # 异步模式下必须有可用调度器，否则后台任务无法继续执行。
                    assert self.mem_scheduler is not None, (
                        "Mem-Scheduler must be working when use asynchronous memory adding."
                    )
                    # 构造一条调度消息，把当前事件及其上下文交给后台调度器。
                    message_item = ScheduleMessageItem(
                        # 调度消息绑定目标用户，便于后台按用户隔离处理。
                        user_id=target_user_id,
                        # 调度消息额外绑定会话 ID，便于异步偏好任务按会话归档。
                        session_id=target_session_id,
                        # 调度消息绑定当前 MemCube，后台任务据此定位记忆库。
                        mem_cube_id=mem_cube_id,
                        # 把这条任务标记为偏好记忆添加任务。
                        label=PREF_ADD_TASK_LABEL,
                        # 把原始消息批次序列化后交给调度器，后台再抽取偏好。
                        content=json.dumps(messages_list),
                        # 记录事件进入调度器的 UTC 时间戳。
                        timestamp=datetime.utcnow(),
                    )
                    # 把调度消息提交给调度器，统一使用列表形式以兼容批量接口。
                    self.mem_scheduler.submit_messages(messages=[message_item])

        # Execute both memory processing functions in parallel
        # 用两个线程并行执行文本记忆和偏好记忆搜索，提高单个 cube 的检索效率。
        with ContextThreadPoolExecutor(max_workers=2) as executor:
            # 提交文本记忆处理任务。
            text_future = executor.submit(process_textual_memory)
            # 提交偏好记忆处理任务。
            pref_future = executor.submit(process_preference_memory)

            # Wait for both tasks to complete
            # 等待文本记忆处理完成；如果子任务抛错，这里会重新抛出。
            text_future.result()
            # 等待偏好记忆处理完成；保证两个写入分支都结束后再继续。
            pref_future.result()

        # user profile
        # 开始组合多个前置条件，只有全部满足才进入核心逻辑。
        if (
            # 只有直接传入单条 memory_content 时，才进入用户画像/文本内容写入分支。
            (memory_content is not None)
            # 确认全局配置开启文本记忆写入。
            and self.config.enable_textual_memory
            # 确认目标 MemCube 存在文本记忆模块。
            and self.mem_cubes[mem_cube_id].text_mem
        ):
            # 只有非 tree_text 后端支持直接更新文本记忆。
            if self.mem_cubes[mem_cube_id].config.text_mem.backend != "tree_text":
                # 为即将写入的文本记忆构造统一元数据。
                metadata = TextualMemoryMetadata(
                    # 元数据记录用户、会话和来源，方便后续检索过滤与溯源。
                    user_id=target_user_id, session_id=target_session_id, source="conversation"
                )
                # 开始一个多行调用或结构声明，后续缩进行会继续补充参数。
                self.mem_cubes[mem_cube_id].text_mem.add(
                    # 把单条文本内容包装成列表形式，以符合 text_mem.add 的批量接口。
                    [TextualMemoryItem(memory=memory_content, metadata=metadata)]
                )
            # 进入备选路径，通常用于默认值或异常情况处理。
            else:
                # 计算并保存 messages_list，供后续逻辑继续使用。
                messages_list = [
                    [{"role": "user", "content": memory_content}]
                ]  # for only user-str input and convert message

                # 调用 mem_reader 从消息或文档中抽取结构化记忆。
                memories = self.mem_reader.get_memory(
                    # 把待抽取的消息批次传入 mem_reader。
                    messages_list,
                    # 声明输入类型是聊天对话。
                    type="chat",
                    # 计算并保存 info，供后续逻辑继续使用。
                    info={"user_id": target_user_id, "session_id": target_session_id},
                    # 异步模式先用 fast 快速抽取；同步模式用 fine 更精细地抽取。
                    mode="fast" if sync_mode == "async" else "fine",
                )

                # 准备收集一批新增记忆 ID。
                mem_ids = []
                # 逐批处理 mem_reader 抽取出的记忆集合。
                for mem in memories:
                    # 把当前批次记忆写入 text_mem，并得到这批新增 ID。
                    mem_id_list: list[str] = self.mem_cubes[mem_cube_id].text_mem.add(mem)
                    # 记录流程状态或诊断信息，方便追踪运行路径和耗时。
                    logger.info(
                        f"Added memory user {target_user_id} to memcube {mem_cube_id}: {mem_id_list}"
                    )
                    # 把当前批次 ID 合并到总 ID 列表中。
                    mem_ids.extend(mem_id_list)

                # submit messages for scheduler
                # 调度器启用且实例可用时，才向后台提交查询/回答/添加等事件。
                if self.enable_mem_scheduler and self.mem_scheduler is not None:
                    # 异步模式下，添加后的进一步处理交给 MEM_READ 调度任务。
                    if sync_mode == "async":
                        # 构造一条调度消息，把当前事件及其上下文交给后台调度器。
                        message_item = ScheduleMessageItem(
                            # 调度消息绑定目标用户，便于后台按用户隔离处理。
                            user_id=target_user_id,
                            # 调度消息绑定当前 MemCube，后台任务据此定位记忆库。
                            mem_cube_id=mem_cube_id,
                            # 把这条任务标记为异步记忆读取/整理任务。
                            label=MEM_READ_TASK_LABEL,
                            # 把新增记忆 ID 序列化成 JSON 字符串作为任务内容。
                            content=json.dumps(mem_ids),
                            # 记录事件进入调度器的 UTC 时间戳。
                            timestamp=datetime.utcnow(),
                        )
                        # 把调度消息提交给调度器，统一使用列表形式以兼容批量接口。
                        self.mem_scheduler.submit_messages(messages=[message_item])
                    # 进入备选路径，通常用于默认值或异常情况处理。
                    else:
                        # 构造一条调度消息，把当前事件及其上下文交给后台调度器。
                        message_item = ScheduleMessageItem(
                            # 调度消息绑定目标用户，便于后台按用户隔离处理。
                            user_id=target_user_id,
                            # 调度消息绑定当前 MemCube，后台任务据此定位记忆库。
                            mem_cube_id=mem_cube_id,
                            # 把这条任务标记为同步添加记忆后的后处理任务。
                            label=ADD_TASK_LABEL,
                            # 把新增记忆 ID 序列化成 JSON 字符串作为任务内容。
                            content=json.dumps(mem_ids),
                            # 记录事件进入调度器的 UTC 时间戳。
                            timestamp=datetime.utcnow(),
                        )
                        # 把调度消息提交给调度器，统一使用列表形式以兼容批量接口。
                        self.mem_scheduler.submit_messages(messages=[message_item])

        # user doc input
        # 开始组合多个前置条件，只有全部满足才进入核心逻辑。
        if (
            # 只有传入文档路径时，才进入文档记忆导入分支。
            (doc_path is not None)
            # 确认全局配置开启文本记忆写入。
            and self.config.enable_textual_memory
            # 确认目标 MemCube 存在文本记忆模块。
            and self.mem_cubes[mem_cube_id].text_mem
        ):
            # 递归扫描文档路径，得到可导入的文件列表。
            documents = self._get_all_documents(doc_path)
            # 调用 mem_reader 从文档文件中抽取记忆。
            doc_memories = self.mem_reader.get_memory(
                # 把待解析的文档路径列表传给 mem_reader。
                documents,
                # 声明输入类型是文档。
                type="doc",
                # 计算并保存 info，供后续逻辑继续使用。
                info={"user_id": target_user_id, "session_id": target_session_id},
            )

            # 准备收集一批新增记忆 ID。
            mem_ids = []
            # 逐批写入从文档中抽取出的记忆。
            for mem in doc_memories:
                # 把当前批次记忆写入 text_mem，并得到这批新增 ID。
                mem_id_list: list[str] = self.mem_cubes[mem_cube_id].text_mem.add(mem)
                # 把当前批次 ID 合并到总 ID 列表中。
                mem_ids.extend(mem_id_list)

            # submit messages for scheduler
            # 调度器启用且实例可用时，才向后台提交查询/回答/添加等事件。
            if self.enable_mem_scheduler and self.mem_scheduler is not None:
                # 构造一条调度消息，把当前事件及其上下文交给后台调度器。
                message_item = ScheduleMessageItem(
                    # 调度消息绑定目标用户，便于后台按用户隔离处理。
                    user_id=target_user_id,
                    # 调度消息绑定当前 MemCube，后台任务据此定位记忆库。
                    mem_cube_id=mem_cube_id,
                    # 把这条任务标记为同步添加记忆后的后处理任务。
                    label=ADD_TASK_LABEL,
                    # 把新增记忆 ID 序列化成 JSON 字符串作为任务内容。
                    content=json.dumps(mem_ids),
                    # 记录事件进入调度器的 UTC 时间戳。
                    timestamp=datetime.utcnow(),
                )
                # 把调度消息提交给调度器，统一使用列表形式以兼容批量接口。
                self.mem_scheduler.submit_messages(messages=[message_item])

        # 所有输入分支处理结束后，记录记忆添加成功。
        logger.info(f"Add memory to {mem_cube_id} successfully")

    # 根据 memory_id 从指定 MemCube 中读取单条记忆。
    def get(
        # 计算并保存 self, mem_cube_id: str, memory_id: str, user_id: str | None，供后续逻辑继续使用。
        self, mem_cube_id: str, memory_id: str, user_id: str | None = None
    ) -> TextualMemoryItem | ActivationMemoryItem | ParametricMemoryItem:
        """
        Get a textual memory from a MemCube.

        Args:
            mem_cube_id (str): The identifier of the MemCube to get the memory from.
            memory_id (str): The identifier of the  memory to get.
            user_id (str, optional): The identifier of the user to get the memory from.
                If None, the default user is used.

        Returns:
            Union[TextualMemoryItem, ActivationMemoryItem, ParametricMemoryItem]: The requested memory item.
        """
        # 确定本次操作的目标用户：显式传入优先，否则使用 MOSCore 默认用户。
        target_user_id = user_id if user_id is not None else self.user_id
        # Validate user has access to this cube
        # 读取/更新/删除前确认用户拥有目标 MemCube 的访问权限。
        self._validate_cube_access(target_user_id, mem_cube_id)
        # 没有指定 MemCube 时，尝试从用户可访问列表中选择默认写入目标。
        if mem_cube_id is None:
            # Try to find a default cube for the user
            # 读取目标用户可访问的 cube，用于未指定 cube_id 时选择默认目标。
            accessible_cubes = self.user_manager.get_user_cubes(target_user_id)
            # 如果用户没有任何可访问 cube，就无法确定默认写入位置。
            if not accessible_cubes:
                # 遇到不可恢复的非法状态时抛出 ValueError，让调用方明确知道输入或配置不满足要求。
                raise ValueError(
                    f"No accessible cubes found for user '{target_user_id}'. Please register a cube first."
                )
            # 当前实现临时使用第一个可访问 cube 作为默认目标，原 TODO 提醒未来应支持更合理选择。
            mem_cube_id = accessible_cubes[0].cube_id  # TODO not only first
        # 进入备选路径，通常用于默认值或异常情况处理。
        else:
            # 读取/更新/删除前确认用户拥有目标 MemCube 的访问权限。
            self._validate_cube_access(target_user_id, mem_cube_id)

        # 确认目标 MemCube 已加载到当前运行时容器。
        assert mem_cube_id in self.mem_cubes, (
            f"MemCube with ID {mem_cube_id} does not exist. please regiester"
        )
        # 从目标文本记忆模块读取指定 ID 的记忆并返回。
        return self.mem_cubes[mem_cube_id].text_mem.get(memory_id)

    # 读取指定 MemCube 中所有可用的文本记忆和激活记忆。
    def get_all(
        # 计算并保存 self, mem_cube_id: str | None，供后续逻辑继续使用。
        self, mem_cube_id: str | None = None, user_id: str | None = None
    ) -> MOSSearchResult:
        """
        Get all textual memories from a MemCube.

        Args:
            mem_cube_id (str, optional): The identifier of the MemCube to get the memories from.
                If None, all MemCube for the user is used.
            user_id (str, optional): The identifier of the user to get the memories from.
                If None, the default user is used.

        Returns:
            MemoryResult: A dictionary containing the search results.
        """
        # 初始化 get_all 的返回结构，包含参数、激活和文本记忆列表。
        result: MOSSearchResult = {"para_mem": [], "act_mem": [], "text_mem": []}
        # 确定本次操作的目标用户：显式传入优先，否则使用 MOSCore 默认用户。
        target_user_id = user_id if user_id is not None else self.user_id
        # Validate user has access to this cube
        # 没有指定 MemCube 时，尝试从用户可访问列表中选择默认写入目标。
        if mem_cube_id is None:
            # Try to find a default cube for the user
            # 读取目标用户可访问的 cube，用于未指定 cube_id 时选择默认目标。
            accessible_cubes = self.user_manager.get_user_cubes(target_user_id)
            # 如果用户没有任何可访问 cube，就无法确定默认写入位置。
            if not accessible_cubes:
                # 遇到不可恢复的非法状态时抛出 ValueError，让调用方明确知道输入或配置不满足要求。
                raise ValueError(
                    f"No accessible cubes found for user '{target_user_id}'. Please register a cube first."
                )
            # 当前实现临时使用第一个可访问 cube 作为默认目标，原 TODO 提醒未来应支持更合理选择。
            mem_cube_id = accessible_cubes[0].cube_id  # TODO not only first
        # 进入备选路径，通常用于默认值或异常情况处理。
        else:
            # 读取/更新/删除前确认用户拥有目标 MemCube 的访问权限。
            self._validate_cube_access(target_user_id, mem_cube_id)
        # 文本记忆功能开启且模块存在时，读取全部文本记忆。
        if self.config.enable_textual_memory and self.mem_cubes[mem_cube_id].text_mem:
            # 把当前 MemCube 的文本记忆全集追加到返回结果中。
            result["text_mem"].append(
                # 把 cube 来源和全部文本记忆绑定在同一个结果项中。
                {"cube_id": mem_cube_id, "memories": self.mem_cubes[mem_cube_id].text_mem.get_all()}
            )
        # 激活记忆功能开启且模块存在时，读取全部激活记忆。
        if self.config.enable_activation_memory and self.mem_cubes[mem_cube_id].act_mem:
            # 把当前 MemCube 的激活记忆全集追加到返回结果中。
            result["act_mem"].append(
                # 把 cube 来源和全部激活记忆绑定在同一个结果项中。
                {"cube_id": mem_cube_id, "memories": self.mem_cubes[mem_cube_id].act_mem.get_all()}
            )
        # 返回按记忆类型组织好的搜索结果。
        return result

    # 更新指定 MemCube 中的一条文本记忆；tree_text 后端目前不支持该操作。
    def update(
        # 当前 MOSCore 实例本身。
        self,
        # 参数 mem_cube_id 参与 update 的业务流程，调用方可通过它改变本次操作的上下文或目标。
        mem_cube_id: str,
        # 参数 memory_id 参与 update 的业务流程，调用方可通过它改变本次操作的上下文或目标。
        memory_id: str,
        # 待写入的新记忆内容，可以是文本记忆对象或字典。
        text_memory_item: TextualMemoryItem | dict[str, Any],
        # 参数 user_id 参与 update 的业务流程，调用方可通过它改变本次操作的上下文或目标。
        user_id: str | None = None,
    ) -> None:
        """
        Update a textual memory in a MemCube by text_memory_id and text_memory_id.

        Args:
            mem_cube_id (str): The identifier of the MemCube to update the memory in.
            memory_id (str): The identifier of the textual memory to update.
            text_memory_item (TextualMemoryItem | dict[str, Any]): The updated textual memory item.
        """
        # 确认目标 MemCube 已加载到当前运行时容器。
        assert mem_cube_id in self.mem_cubes, (
            f"MemCube with ID {mem_cube_id} does not exist. please regiester"
        )
        # 确定本次操作的目标用户：显式传入优先，否则使用 MOSCore 默认用户。
        target_user_id = user_id if user_id is not None else self.user_id
        # Validate user has access to this cube
        # 读取/更新/删除前确认用户拥有目标 MemCube 的访问权限。
        self._validate_cube_access(target_user_id, mem_cube_id)
        # 没有指定 MemCube 时，尝试从用户可访问列表中选择默认写入目标。
        if mem_cube_id is None:
            # Try to find a default cube for the user
            # 读取目标用户可访问的 cube，用于未指定 cube_id 时选择默认目标。
            accessible_cubes = self.user_manager.get_user_cubes(target_user_id)
            # 如果用户没有任何可访问 cube，就无法确定默认写入位置。
            if not accessible_cubes:
                # 遇到不可恢复的非法状态时抛出 ValueError，让调用方明确知道输入或配置不满足要求。
                raise ValueError(
                    f"No accessible cubes found for user '{target_user_id}'. Please register a cube first."
                )
            # 当前实现临时使用第一个可访问 cube 作为默认目标，原 TODO 提醒未来应支持更合理选择。
            mem_cube_id = accessible_cubes[0].cube_id  # TODO not only first
        # 进入备选路径，通常用于默认值或异常情况处理。
        else:
            # 读取/更新/删除前确认用户拥有目标 MemCube 的访问权限。
            self._validate_cube_access(target_user_id, mem_cube_id)
        # 只有非 tree_text 后端支持直接更新文本记忆。
        if self.mem_cubes[mem_cube_id].config.text_mem.backend != "tree_text":
            # 调用底层文本记忆模块更新指定 memory_id 的内容。
            self.mem_cubes[mem_cube_id].text_mem.update(memory_id, memories=text_memory_item)
            # 记录指定记忆更新成功。
            logger.info(f"MemCube {mem_cube_id} updated memory {memory_id}")
        # 进入备选路径，通常用于默认值或异常情况处理。
        else:
            # 记录警告日志，表示系统进入非理想但可继续的路径。
            logger.warning(
                f" {self.mem_cubes[mem_cube_id].config.text_mem.backend} does not support update memory"
            )

    # 删除指定 MemCube 中的一条文本记忆。
    def delete(self, mem_cube_id: str, memory_id: str, user_id: str | None = None) -> None:
        """
        Delete a textual memory from a MemCube by memory_id.

        Args:
            mem_cube_id (str): The identifier of the MemCube to delete the memory from.
            memory_id (str): The identifier of the  memory to delete.
        """
        # 确认目标 MemCube 已加载到当前运行时容器。
        assert mem_cube_id in self.mem_cubes, (
            f"MemCube with ID {mem_cube_id} does not exist. please regiester"
        )
        # 确定本次操作的目标用户：显式传入优先，否则使用 MOSCore 默认用户。
        target_user_id = user_id if user_id is not None else self.user_id
        # Validate user has access to this cube
        # 读取/更新/删除前确认用户拥有目标 MemCube 的访问权限。
        self._validate_cube_access(target_user_id, mem_cube_id)
        # 没有指定 MemCube 时，尝试从用户可访问列表中选择默认写入目标。
        if mem_cube_id is None:
            # Try to find a default cube for the user
            # 读取目标用户可访问的 cube，用于未指定 cube_id 时选择默认目标。
            accessible_cubes = self.user_manager.get_user_cubes(target_user_id)
            # 如果用户没有任何可访问 cube，就无法确定默认写入位置。
            if not accessible_cubes:
                # 遇到不可恢复的非法状态时抛出 ValueError，让调用方明确知道输入或配置不满足要求。
                raise ValueError(
                    f"No accessible cubes found for user '{target_user_id}'. Please register a cube first."
                )
            # 当前实现临时使用第一个可访问 cube 作为默认目标，原 TODO 提醒未来应支持更合理选择。
            mem_cube_id = accessible_cubes[0].cube_id  # TODO not only first
        # 进入备选路径，通常用于默认值或异常情况处理。
        else:
            # 读取/更新/删除前确认用户拥有目标 MemCube 的访问权限。
            self._validate_cube_access(target_user_id, mem_cube_id)
        # 调用底层文本记忆模块删除指定 memory_id。
        self.mem_cubes[mem_cube_id].text_mem.delete(memory_id)
        # 记录指定记忆删除成功。
        logger.info(f"MemCube {mem_cube_id} deleted memory {memory_id}")

    # 删除指定 MemCube 中的全部文本记忆。
    def delete_all(self, mem_cube_id: str | None = None, user_id: str | None = None) -> None:
        """
        Delete all textual memories from a MemCube for user.

        Args:
            mem_cube_id (str): The identifier of the MemCube to delete the memories from.
        """
        # 确认目标 MemCube 已加载到当前运行时容器。
        assert mem_cube_id in self.mem_cubes, (
            f"MemCube with ID {mem_cube_id} does not exist. please regiester"
        )
        # 确定本次操作的目标用户：显式传入优先，否则使用 MOSCore 默认用户。
        target_user_id = user_id if user_id is not None else self.user_id
        # Validate user has access to this cube
        # 读取/更新/删除前确认用户拥有目标 MemCube 的访问权限。
        self._validate_cube_access(target_user_id, mem_cube_id)
        # 没有指定 MemCube 时，尝试从用户可访问列表中选择默认写入目标。
        if mem_cube_id is None:
            # Try to find a default cube for the user
            # 读取目标用户可访问的 cube，用于未指定 cube_id 时选择默认目标。
            accessible_cubes = self.user_manager.get_user_cubes(target_user_id)
            # 如果用户没有任何可访问 cube，就无法确定默认写入位置。
            if not accessible_cubes:
                # 遇到不可恢复的非法状态时抛出 ValueError，让调用方明确知道输入或配置不满足要求。
                raise ValueError(
                    f"No accessible cubes found for user '{target_user_id}'. Please register a cube first."
                )
            # 当前实现临时使用第一个可访问 cube 作为默认目标，原 TODO 提醒未来应支持更合理选择。
            mem_cube_id = accessible_cubes[0].cube_id  # TODO not only first
        # 进入备选路径，通常用于默认值或异常情况处理。
        else:
            # 读取/更新/删除前确认用户拥有目标 MemCube 的访问权限。
            self._validate_cube_access(target_user_id, mem_cube_id)
        # 调用底层文本记忆模块清空全部文本记忆。
        self.mem_cubes[mem_cube_id].text_mem.delete_all()
        # 记录指定 MemCube 的文本记忆已清空。
        logger.info(f"MemCube {mem_cube_id} deleted all memories")

    # 把已加载的 MemCube 持久化导出到目标目录。
    def dump(
        # 计算并保存 self, dump_dir: str, user_id: str | None，供后续逻辑继续使用。
        self, dump_dir: str, user_id: str | None = None, mem_cube_id: str | None = None
    ) -> None:
        """Dump the MemCube to a dictionary.
        Args:
            dump_dir (str): The directory to dump the MemCube to.
            user_id (str, optional): The identifier of the user to dump the MemCube from.
                If None, the default user is used.
            mem_cube_id (str, optional): The identifier of the MemCube to dump.
                If None, the default MemCube for the user is used.
        """
        # 确定本次操作的目标用户：显式传入优先，否则使用 MOSCore 默认用户。
        target_user_id = user_id if user_id is not None else self.user_id
        # 读取目标用户可访问的 cube，用于未指定 cube_id 时选择默认目标。
        accessible_cubes = self.user_manager.get_user_cubes(target_user_id)
        # 如果没有显式传入 cube_id，就从可访问列表中选择默认 cube。
        if not mem_cube_id:
            # 使用第一个可访问 cube 作为默认导出/加载目标。
            mem_cube_id = accessible_cubes[0].cube_id
        # 权限存在不代表运行时已加载；这里确认目标 cube 已在内存中。
        if mem_cube_id not in self.mem_cubes:
            # 抛出异常，向调用方明确报告当前操作无法继续。
            raise ValueError(f"MemCube with ID {mem_cube_id} does not exist. please regiester")
        # 调用 MemCube 自身的 dump 方法，把数据导出到指定目录。
        self.mem_cubes[mem_cube_id].dump(dump_dir)
        # 记录 MemCube 导出位置。
        logger.info(f"MemCube {mem_cube_id} dumped to {dump_dir}")

    # 从目标目录把 MemCube 数据加载回当前运行时对象。
    def load(
        # 当前 MOSCore 实例本身。
        self,
        # MemCube 加载的来源目录。
        load_dir: str,
        # 参数 user_id 参与 load 的业务流程，调用方可通过它改变本次操作的上下文或目标。
        user_id: str | None = None,
        # 可选 MemCube ID；为空时通常根据输入或用户权限推导。
        mem_cube_id: str | None = None,
        # 可选指定加载哪些类型的记忆；为空时由底层 load 默认处理。
        memory_types: list[Literal["text_mem", "act_mem", "para_mem", "pref_mem"]] | None = None,
    ) -> None:
        """Dump the MemCube to a dictionary.
        Args:
            load_dir (str): The directory to load the MemCube from.
            user_id (str, optional): The identifier of the user to load the MemCube from.
                If None, the default user is used.
            mem_cube_id (str, optional): The identifier of the MemCube to load.
                If None, the default MemCube for the user is used.
        """
        # 确定本次操作的目标用户：显式传入优先，否则使用 MOSCore 默认用户。
        target_user_id = user_id if user_id is not None else self.user_id
        # 读取目标用户可访问的 cube，用于未指定 cube_id 时选择默认目标。
        accessible_cubes = self.user_manager.get_user_cubes(target_user_id)
        # 如果没有显式传入 cube_id，就从可访问列表中选择默认 cube。
        if not mem_cube_id:
            # 使用第一个可访问 cube 作为默认导出/加载目标。
            mem_cube_id = accessible_cubes[0].cube_id
        # 权限存在不代表运行时已加载；这里确认目标 cube 已在内存中。
        if mem_cube_id not in self.mem_cubes:
            # 抛出异常，向调用方明确报告当前操作无法继续。
            raise ValueError(f"MemCube with ID {mem_cube_id} does not exist. please regiester")
        # 调用 MemCube 自身的 load 方法，从目录加载指定类型的记忆数据。
        self.mem_cubes[mem_cube_id].load(load_dir, memory_types=memory_types)
        # 记录 MemCube 加载来源。
        logger.info(f"MemCube {mem_cube_id} loaded from {load_dir}")

    # 返回当前用户的基础信息以及其可访问 MemCube 的加载状态。
    def get_user_info(self) -> dict[str, Any]:
        """Get current user information including accessible cubes.
        TODO: maybe input user_id
        Returns:
            dict: User information and accessible cubes.
        """
        # 读取当前默认用户对象，用于构造用户信息响应。
        user = self.user_manager.get_user(self.user_id)
        # 如果当前用户不存在，直接返回空字典表示无法提供信息。
        if not user:
            # 返回空结果，避免后续访问 None 对象属性。
            return {}

        # 读取当前用户可访问的全部 MemCube。
        accessible_cubes = self.user_manager.get_user_cubes(self.user_id)

        # 返回当前方法处理后的结果。
        return {
            # 输出用户 ID。
            "user_id": user.user_id,
            # 输出用户名。
            "user_name": user.user_name,
            # 兼容角色可能是枚举或普通值两种情况。
            "role": user.role.value if hasattr(user.role, "value") else user.role,
            # 把创建时间转换为 ISO 字符串，便于 JSON 序列化。
            "created_at": user.created_at.isoformat(),
            # 开始构造当前用户可访问 MemCube 的详情列表。
            "accessible_cubes": [
                # 开始构造一个结构化字典或映射对象。
                {
                    # 输出 MemCube ID。
                    "cube_id": cube.cube_id,
                    # 输出 MemCube 名称。
                    "cube_name": cube.cube_name,
                    # 输出 MemCube 的存储或来源路径。
                    "cube_path": cube.cube_path,
                    # 输出 MemCube 所有者用户 ID。
                    "owner_id": cube.owner_id,
                    # 标记该 MemCube 当前是否已经加载到运行时容器。
                    "is_loaded": cube.cube_id in self.mem_cubes,
                }
                # 遍历当前用户可访问的所有 cube，逐个构造详情。
                for cube in accessible_cubes
            ],
        }

    # 把当前用户有权访问的 MemCube 授权给另一个已存在用户。
    def share_cube_with_user(self, cube_id: str, target_user_id: str) -> bool:
        """Share a cube with another user.

        Args:
            cube_id (str): The cube ID to share.
            target_user_id (str): The user ID to share with.

        Returns:
            bool: True if successful, False otherwise.
        """
        # Validate current user has access to this cube
        # 分享前先确认当前用户自己有权访问该 MemCube。
        self._validate_cube_access(self.user_id, cube_id)

        # Validate target user exists
        # 确认目标用户存在且处于激活状态，否则不能授权。
        if not self.user_manager.validate_user(target_user_id):
            # 抛出异常，向调用方明确报告当前操作无法继续。
            raise ValueError(f"Target user '{target_user_id}' does not exist or is inactive.")

        # 委托用户管理器建立目标用户与 MemCube 的访问关系，并返回是否成功。
        return self.user_manager.add_user_to_cube(target_user_id, cube_id)

    # 根据历史对话判断当前查询是否需要改写，并返回改写后的查询。
    def get_query_rewrite(self, query: str, user_id: str | None = None):
        """
        Rewrite user's query according the context.
        Args:
            query (str): The search query that needs rewriting.
            user_id(str, optional): The identifier of the user that the query belongs to.
                If None, the default user is used.

        Returns:
            str: query after rewriting process.
        """
        # 确定本次操作的目标用户：显式传入优先，否则使用 MOSCore 默认用户。
        target_user_id = user_id if user_id is not None else self.user_id
        # 取出目标用户历史对话，作为查询改写的上下文来源。
        chat_history = self.chat_history_manager[target_user_id]

        # 把历史对话拼成带分隔符的字符串，交给提示词模板使用。
        dialogue = "————{}".format("\n————".join(chat_history.chat_history))
        # 把历史对话和当前查询填入查询改写模板。
        user_prompt = QUERY_REWRITING_PROMPT.format(dialogue=dialogue, query=query)
        # 把改写提示包装成用户消息格式，交给聊天模型生成。
        messages = {"role": "user", "content": user_prompt}
        # 调用聊天模型执行查询改写判断和生成。
        rewritten_result = self.chat_llm.generate(messages=messages)
        # 把模型返回的 JSON 字符串解析成字典，便于读取字段。
        rewritten_result = json.loads(rewritten_result)
        # 如果模型判断当前问题与历史对话有关，就优先使用改写后的问题。
        if rewritten_result.get("former_dialogue_related", False):
            # 取出模型生成的改写问题。
            rewritten_query = rewritten_result.get("rewritten_question")
            # 如果改写结果非空则返回它，否则回退到原始查询。
            return rewritten_query if len(rewritten_query) > 0 else query
        # 如果当前查询与历史对话无关，保持原始查询不变。
        return query
