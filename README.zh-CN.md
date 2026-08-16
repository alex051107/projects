# 证据驱动的 Agent 与 Scientific ML 项目集

这个公开仓库集中展示目前四条核心项目线：EvidenceOps、CarePlan、
Dynamics Atlas，以及 HSP90/LiGaMD。它们共同强调的不是“让模型自主做完
一切”，而是模型周围的工程控制层：证据契约、确定性校验、持久状态、人类
决策权、失败恢复，以及可审计的 claim ceiling。

[English README](README.md)

## 四个核心项目

| 项目 | 公开入口 | 已实现内容 | 不能外推成什么 |
| --- | --- | --- | --- |
| **EvidenceOps / EvidenceUp** | [Public MVP](https://github.com/alex051107/evidenceops-public-mvp) | 公开来源登记、解析与切块、带 citation 的检索、小 schema 抽取、风险检查、小型评测集、失败分析和静态 Evidence Console。 | 静态网页不等于动态后端已生产部署，也不证明真实客户使用或生产规模。 |
| **CarePlan** | [`careplan_workflow_harness/`](careplan_workflow_harness/) | 仅 synthetic 输入、确定性 hard stop、SQLite 幂等、严格 draft schema、fail-closed 校验、乐观并发复核，以及只有 pharmacist 角色才能批准的人工闸门。 | 不是临床系统，不处理真实患者数据，不提供医疗建议，模型不能批准计划。 |
| **Dynamics Atlas** | [`scientific_evidence_harness/`](scientific_evidence_harness/) | source identity、mapping coverage、measurement semantics、maturity state 和 claim ceiling 校验。 | synthetic fixture 只证明治理逻辑可运行，不证明生物学结论、跨系统泛化或 Agent effectiveness。 |
| **HSP90 / LiGaMD experimental-pKoff** | [`trajectory_event_harness/`](trajectory_event_harness/) · [公开 toolkit](https://github.com/alex051107/ligamd-pkoff-toolkit) | replica-aware 的持续事件、recapture、right censoring 与 provenance；另有公开轨迹特征和 experimental-pKoff 工具。 | experimental `pKoff` 是实验 assay 标签，不是由模拟时间直接得到的 physical `koff`；当前结果也不是已选定的最终科学模型。 |

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
属于较早期 prototype，不是当前四项目证据包。其 roadmap 不应被当成所有模型、
数据源、dashboard 或端到端运行都已完成的证明。
