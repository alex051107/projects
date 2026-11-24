# 刘振鹏 | 项目作品集

我做 AI Agent 和科学软件，把模型输出变成可以检查、测试和复现的结果。

[English README](README.md)

## 简历中的五个项目

| 项目 | 从哪里看起 |
| --- | --- |
| **Dynamics Atlas** · 科学 Agent | [运行时与决策流程](https://github.com/alex051107/dynamics-atlas-harness)校验分析提案和工具输出；[文献图谱与证据流程](https://github.com/alex051107/dynamics-atlas-relaxdb/blob/main/PORTFOLIO.md)把文献证据整理为可追溯的蛋白质记录。[本地校验示例](scientific_evidence_harness/)演示证据不足时如何限制结论。 |
| **TaskFactory** · 多 Agent 评测 | [引擎与架构](https://github.com/alex051107/agent-taskfactory-public)。用 LangGraph 串联评测规则、评分器设计、盲审、任务冻结和模型试跑。 |
| **EvidenceOps** · 科学证据检索 | [公开管线](https://github.com/alex051107/evidenceops-public-mvp) · [证据台](https://alex051107.github.io/evidenceops-public-mvp/)。从检索到结构化提取和复核，保留来源与引文。 |
| **CarePlan** · 受控起草与人工复核 | [公开工作流](careplan_workflow_harness/)。用模拟输入展示资格规则在哪里拦截，以及哪些决策由人负责。 |
| **HSP90 LiGaMD3** · 分子模拟与科学机器学习 | [可复现工具包](https://github.com/alex051107/ligamd-pkoff-toolkit) · [轨迹事件识别](trajectory_event_harness/)。把重复轨迹的事件检测和特征提取，与实验标签的预测分开。这里的 `pKoff` 来自实验标签，不是由模拟时间直接估算的物理解离速率。 |

## 项目案例

- **[Legal Benchmark QA](portfolio/legal-benchmark-qa/)**：用一份虚构合同的两个版本演示盲审、评分和返修，区分模型答错与评测题缺少材料。
- **[TrpB Path Planning](portfolio/trpb-path-planning/)**：比较直接插值与经过部分闭合构象的两条候选路径，说明何时修改或暂停。
- **[Benchmark QA Demo](portfolio/benchmark-qa-demo/)**：用合成样例运行离线检查，查看输出。

## 配套控制层

[`career_agent_harness/`](career_agent_harness/) 把同一套思路用于人工控制的
求职流程：来源时效、证据化 readiness、容量约束、外部动作前确认，以及只按
allowlist 导出的公开摘要。它不会自己投递、发消息、约时间或填写结果。

## 本地验证

```bash
python career_agent_harness/scripts/test_harness.py
python -m unittest discover -s careplan_workflow_harness/tests -v
python -m unittest discover -s scientific_evidence_harness/tests -v
python -m unittest discover -s trajectory_event_harness/tests -v
```

这些测试只验证对应的软件契约。测试通过不等于云端部署、真实用户效果、临床
效果、科学外部验证或 Agent 增量价值已经成立。

## 公开边界

仓库不包含简历与头像、联系方式、申请记录、原始会议材料、密钥或凭据、真实
患者/客户数据、未公开 assay 表、原始轨迹与 topology、合作者记录或 campaign
身份对照。公开示例只使用 synthetic fixture 或明确可公开的来源。

仓库里保留的 bee forecasting、water-level forecasting 和 virtual screening
属于较早期原型，不在上面的简历项目列表中。其路线图不应被当成所有模型、
数据源、dashboard 或端到端运行都已完成的证明。
