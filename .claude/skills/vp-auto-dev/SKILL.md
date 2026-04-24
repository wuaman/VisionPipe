---
name: vp-auto-dev
description: |
  VisionPipe-py 规范驱动开发代理，自动完成「读规范→找任务→编码→测试→持久化」全流程。
  
  触发场景：
  - 用户说 "vp auto dev"、"自动开发"、"实现任务"、"下一个任务"
  - 用户指定具体任务ID，如 "vp dev T0.1"、"实现 T1.3"
  - 用户说 "开始开发"、"继续开发"
---

# VisionPipe-py Auto-Dev Skill

## 概述

本 skill 用于 VisionPipe-py 项目的自动化开发流程。基于 `DEV_SPEC.md` 规范文件，自动查找待实现任务、编写代码、运行测试、更新状态。

## 工作流程

### Step 1: 同步规范 (Sync Spec)

首先运行同步脚本，将 `DEV_SPEC.md` 拆分为章节文件：

```bash
python .claude/skills/vp-auto-dev/scripts/sync_spec.py
```

这会将规范拆分到 `references/` 目录：
- `00-dev-env.md` - 开发环境
- `01-overview.md` - 项目概述
- `02-features.md` - 核心特点
- `03-tech-stack.md` - 技术选型
- `04-testing.md` - 测试方案
- `05-architecture.md` - 系统架构
- `06-schedule.md` - 项目排期（任务详情）
- `07-dev-spec.md` - 开发规范

### Step 2: 查找任务 (Find Task)

从「6.2 项目跟踪表」中查找任务：

1. **优先级**：先找状态为 `[~]`（进行中）的任务
2. **其次**：找第一个状态为 `[ ]`（待开始）的任务
3. **指定任务**：如果用户指定了任务ID（如 T0.1），则定位到该任务

任务状态标记：
| 标记 | 状态 |
|------|------|
| `[ ]` | 待开始 |
| `[~]` | 进行中 |
| `[x]` | 已完成 |

### Step 3: 实现代码 (Implement)

根据任务描述，读取相关规范文件：

1. **必读**：`06-schedule.md`（任务详情）
2. **按需读取**：
   - 涉及架构设计 → `05-architecture.md`
   - 涉及测试 → `04-testing.md`
   - 涉及技术选型 → `03-tech-stack.md`
   - 涉及开发规范 → `07-dev-spec.md`

从任务描述中提取：
- 要创建/修改的文件列表
- 要实现的类/函数
- 验收标准
- 测试方法

然后编写代码实现。

### Step 3.5: 独立测试编写 (Independent Test)

**核心原则**：实现者不写测试。测试由独立 Agent 完成，该 Agent 只能看到**接口定义**和**规范**，不能看到实现细节。

根据任务涉及的语言，派生对应的测试 Agent（可同时派生两个并行执行）：

---

#### C++ 测试 Agent

```
Agent(
  description: "为 <任务ID> 编写 C++ 独立测试",
  prompt: """
你是一个独立测试工程师，负责为刚刚实现的 C++ 模块编写测试用例。

## 你的信息来源（仅限以下内容）
- 任务规范：<粘贴 06-schedule.md 中该任务的完整描述>
- 验收标准：<粘贴任务的验收标准>
- 头文件（接口）：<列出本任务涉及的所有 .h 文件路径，让 Agent 自行读取>

## 你不能读取的文件
- 任何 .cpp 实现文件（不要读，不要参考）

## 测试要求
对每个公开接口，必须覆盖：
1. 正常路径（happy path）
2. 边界值（空、零、最大容量、单元素）
3. 错误/异常路径（非法参数、停止状态、重复操作）
4. 状态转换序列（INIT→RUNNING→STOPPED 等）
5. 并发场景（如适用）

## 约束
- 断言必须精确：用 EXPECT_EQ 而非 EXPECT_GE，除非确实允许范围
- 禁止 catch(...) 吞异常：用 EXPECT_THROW 或 ASSERT_NO_THROW
- 不要为了通过测试而降低断言强度

## 输出
将测试代码写入 tests/unit/cpp/ 下对应文件（追加或新建）。
"""
)
```

---

#### Python 测试 Agent

仅当任务涉及 Python 层（nanobind 绑定、Python DSL、管理 API）时派生：

```
Agent(
  description: "为 <任务ID> 编写 Python 独立测试",
  prompt: """
你是一个独立测试工程师，负责为刚刚实现的 Python 模块编写测试用例。

## 你的信息来源（仅限以下内容）
- 任务规范：<粘贴 06-schedule.md 中该任务的完整描述>
- 验收标准：<粘贴任务的验收标准>
- Python 接口定义：<列出以下路径，让 Agent 自行读取>
  - .pyi 类型存根文件（若存在）
  - python/ 下各模块的 __init__.py（仅看导出符号，不看实现）
  - nanobind 绑定的头文件 .h（了解暴露的 C++ 接口）

## 你不能读取的文件
- 任何 Python 实现文件（bind_*.cpp、具体业务逻辑 .py）
- C++ .cpp 实现文件

## 测试要求
对每个公开接口，必须覆盖：
1. 正常路径（happy path）
2. 边界值（None、空列表、零值、极大值）
3. 类型错误与参数错误（传错类型、缺少必填参数）
4. 异常路径（操作顺序错误、资源未初始化时调用）
5. Python ↔ C++ 异常穿透（VisionPipeError 子类能否正确传递到 Python 层）

## 约束
- 断言必须精确：用 assert result == expected，不用 assert result >= expected
- 禁止裸 except 吞异常：用 pytest.raises(SpecificError)，不用 try/except: pass
- 不要为了通过测试而降低断言强度

## 输出
将测试代码写入 tests/unit/python/ 下对应文件（追加或新建）。
"""
)
```

---

两个 Agent 完成后，主 Agent **审查新增测试是否覆盖了边界和负面路径**，若覆盖不足则补充 prompt 让对应 Agent 继续完善。

### Step 4: 测试验证 (Test)

VisionPipe-py 是 C++ + Python 混合项目，测试分两类：

**C++ 测试（Google Test）：**
```bash
cmake -B build && cmake --build build
ctest --test-dir build -R <test_name>
```

**Python 测试（pytest）：**
```bash
uv run pytest tests/unit/python/<test_file>.py -v
```

**自修复循环**（最多 3 轮）：
- 轮次 0-2：运行测试，若失败则分析错误并修复
- 轮次 3 仍失败：停止，向用户报告失败原因

**重要**：所有测试都需要真实 GPU 环境。

### Step 5: 持久化 (Persist)

任务通过测试后：

1. 更新 `DEV_SPEC.md` 中对应任务的状态：
   - `[ ]` → `[x]`（已完成）
   
2. 重新同步规范：
   ```bash
   python3 .claude/skills/vp-auto-dev/scripts/sync_spec.py --force
   ```

3. **Git 提交**：询问用户是否进行 git commit
   - 若用户同意，自动生成规范的 commit 信息并提交
   - 若用户拒绝，跳过此步骤

4. 显示摘要，询问用户是否继续下一任务。

#### Git Commit 信息格式

自动生成符合 [Conventional Commits](https://www.conventionalcommits.org/) 规范的中文提交信息：

```
<type>(<scope>): <中文描述>

- 实现细节 1
- 实现细节 2

Task: T<任务ID>
```

**Type 类型映射**：
| 任务类型 | Commit Type |
|----------|-------------|
| CMake/构建配置 | `build` |
| 新增功能/节点 | `feat` |
| Bug 修复 | `fix` |
| 测试代码 | `test` |
| 文档 | `docs` |
| 代码风格/重构 | `style` / `refactor` |

**示例**：
```
build(cmake): 添加项目骨架和 CMake 配置

- 添加根目录 CMakeLists.txt，支持 CUDA/TensorRT
- 添加 src/core/CMakeLists.txt 核心库配置
- 添加 tests/CMakeLists.txt 集成 Google Test
- 配置 FetchContent 拉取 spdlog/nlohmann-json/nanobind

Task: T0.1
```

```
feat(core): 实现带溢出策略的 BoundedQueue

- 添加 BoundedQueue<T> 模板类
- 支持 DROP_OLDEST / DROP_NEWEST / BLOCK 三种溢出策略
- 添加 QueueStats 用于监控队列状态
- 添加单元测试覆盖所有溢出场景

Task: T0.2
```

#### Git Commit 执行流程

1. **检查变更**：运行 `git status` 和 `git diff` 查看变更内容
2. **生成信息**：根据任务描述和实际修改生成中文 commit 信息
3. **询问确认**：向用户展示生成的 commit 信息，询问是否提交
4. **执行提交**：
   ```bash
   git add <相关文件>
   git commit -m "<生成的commit信息>"
   ```
5. **可选推送**：询问用户是否推送到远程仓库

## 任务依赖处理

任务表中的「依赖」列定义了任务的前置条件：
- 前置任务状态为 `[x]` 时，当前任务才能开始
- 若前置任务未完成，提示用户先完成依赖任务

## 代码风格

提交代码前确保：
- **C++**：`clang-format -i src/**/*.h src/**/*.cpp`
- **Python**：`uv run ruff format python/ && uv run mypy python/`

## 复杂任务处理

对于较大的任务（如涉及多文件修改），可以：
1. 先实现核心逻辑
2. 再补充测试代码
3. 分步验证，确保每个增量可测试

## 输出格式

完成任务后，输出结构化摘要：

```markdown
## 任务完成报告

**任务 ID**: T0.1
**任务名称**: 目录结构与 CMake 配置

### 修改文件
- `CMakeLists.txt`（新建）
- `src/core/CMakeLists.txt`（新建）
- `tests/CMakeLists.txt`（新建）

### 测试结果
- `ctest --test-dir build`: 0 passed, 0 failed

### 验收状态
- [x] cmake -B build && cmake --build build 零错误
- [x] ctest --test-dir build 运行空测试套件，0 failed

### Git 状态
- 是否提交：是/否/待确认
- Commit 信息：`build(cmake): add project skeleton and CMake configuration`
```

## 注意事项

1. **GPU 环境**：确保开发机有 NVIDIA GPU 且 CUDA 正常工作
2. **构建依赖**：首次构建需等待 FetchContent 下载依赖
3. **测试数据**：部分测试需要 `tests/data/` 下的测试资源