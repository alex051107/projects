#!/usr/bin/env python3
"""Policy harness for a human-controlled career operations agent."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


SKILL_ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = SKILL_ROOT / "assets"
TEMPLATE_ROOT = ASSET_ROOT / "state-template"
BUILTIN_PLAN = ASSET_ROOT / "weekly-plan.json"
STATE_FILES: dict[str, type] = {
    "config.json": dict,
    "jobs.json": list,
    "contacts.json": list,
    "events.json": list,
    "activity.json": list,
}
READINESS_LEVELS = ("S0", "S1", "S2", "S3")
OPEN_EVENT_STATUSES = {"open", "pending", "scheduled"}
PRE_APPLICATION_STATUSES = {"discovered", "shortlisted", "ready", "draft"}
ACTIVE_PROCESS_STATUSES = {"oa", "interview", "final"}
SUBMITTED_STATUSES = {"applied", "oa", "interview", "final", "offer", "accepted", "rejected", "withdrawn"}
PUBLIC_CATEGORY_LABELS = {
    "event": "处理 72 小时内到期事项",
    "verified-market-deadline": "人工确认并处理已核验临近岗位窗口",
    "market-reverify": "核验需要重新确认的岗位线索",
    "overdue-next-action": "清理逾期下一步",
    "weekly-artifact": "推进本周可审查产物",
    "market-application": "人工复核 1 个合格岗位",
    "market-monitoring": "监测并核验新岗位",
    "oa-practice": "完成 OA/面试训练块",
    "connection": "完成 1 个有理由的关系动作",
}


class RefreshError(RuntimeError):
    """Expected state or plan validation error."""


def read_json(path: Path, expected_type: type) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except FileNotFoundError as exc:
        raise RefreshError(f"missing required file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RefreshError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, expected_type):
        raise RefreshError(f"{path} must contain a top-level {expected_type.__name__}")
    return value


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def initialize_state(root: Path) -> list[Path]:
    root.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    for filename in STATE_FILES:
        source = TEMPLATE_ROOT / filename
        destination = root / filename
        if not destination.exists():
            shutil.copyfile(source, destination)
            created.append(destination)
    plan_destination = root / "weekly-plan.json"
    if not plan_destination.exists():
        if not BUILTIN_PLAN.exists():
            raise RefreshError(f"bundled plan is missing: {BUILTIN_PLAN}")
        shutil.copyfile(BUILTIN_PLAN, plan_destination)
        created.append(plan_destination)
    (root / "daily").mkdir(exist_ok=True)
    (root / "exports").mkdir(exist_ok=True)
    return created


def parse_iso_date(value: Any, field: str) -> date:
    if not isinstance(value, str):
        raise RefreshError(f"{field} must be an ISO date string")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise RefreshError(f"{field} must be YYYY-MM-DD: {value!r}") from exc


def parse_due(value: Any, timezone: ZoneInfo, field: str) -> datetime | None:
    if value in (None, ""):
        return None
    if not isinstance(value, str):
        raise RefreshError(f"{field} must be an ISO date or date-time")
    try:
        if len(value) == 10:
            return datetime.combine(date.fromisoformat(value), time(23, 59, 59), tzinfo=timezone)
        normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise RefreshError(f"{field} has invalid ISO value: {value!r}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone)
    return parsed.astimezone(timezone)


def choose_week(plan: dict[str, Any], today: date) -> tuple[dict[str, Any], str | None]:
    weeks = plan.get("weeks")
    if not isinstance(weeks, list) or not weeks:
        raise RefreshError("weekly plan must contain a nonempty weeks array")
    parsed: list[tuple[date, date, dict[str, Any]]] = []
    seen: set[str] = set()
    for index, week in enumerate(weeks):
        if not isinstance(week, dict):
            raise RefreshError(f"weeks[{index}] must be an object")
        week_id = week.get("id")
        if not isinstance(week_id, str) or not week_id or week_id in seen:
            raise RefreshError(f"weeks[{index}].id must be unique and nonempty")
        seen.add(week_id)
        start = parse_iso_date(week.get("start"), f"weeks[{index}].start")
        end = parse_iso_date(week.get("end"), f"weeks[{index}].end")
        if end < start:
            raise RefreshError(f"{week_id} ends before it starts")
        parsed.append((start, end, week))
    parsed.sort(key=lambda item: item[0])
    for start, end, week in parsed:
        if start <= today <= end:
            return week, None
    if today < parsed[0][0]:
        return parsed[0][2], "date is before the plan; showing the first week"
    return parsed[-1][2], "date is after the plan; showing the final calendar week"


def verified_readiness(config: dict[str, Any]) -> tuple[str | None, dict[str, int]]:
    records = config.get("readinessEvidence", {})
    if not isinstance(records, dict):
        raise RefreshError("config.readinessEvidence must be an object")
    highest: str | None = None
    counts: dict[str, int] = {}
    ladder_broken = False
    for level in READINESS_LEVELS:
        record = records.get(level, {})
        if not isinstance(record, dict):
            raise RefreshError(f"config.readinessEvidence.{level} must be an object")
        evidence = record.get("evidence", [])
        if not isinstance(evidence, list):
            raise RefreshError(f"config.readinessEvidence.{level}.evidence must be an array")
        counts[level] = len(evidence)
        qualifies = record.get("status") == "verified" and bool(evidence)
        if not ladder_broken and qualifies:
            highest = level
        else:
            ladder_broken = True
    return highest, counts


def week_bounds(week: dict[str, Any]) -> tuple[date, date]:
    return (
        parse_iso_date(week.get("start"), f"{week.get('id', 'week')}.start"),
        parse_iso_date(week.get("end"), f"{week.get('id', 'week')}.end"),
    )


def activity_in_week(activity: list[dict[str, Any]], start: date, end: date) -> list[dict[str, Any]]:
    result = []
    for index, item in enumerate(activity):
        if not isinstance(item, dict):
            raise RefreshError(f"activity[{index}] must be an object")
        raw = item.get("date")
        if not raw:
            continue
        item_date = parse_iso_date(raw, f"activity[{index}].date")
        if start <= item_date <= end:
            result.append(item)
    return result


def open_events_within(
    events: list[dict[str, Any]],
    now: datetime,
    timezone: ZoneInfo,
    hours: int,
) -> list[tuple[datetime, dict[str, Any]]]:
    boundary = now + timedelta(hours=hours)
    result: list[tuple[datetime, dict[str, Any]]] = []
    for index, item in enumerate(events):
        if not isinstance(item, dict):
            raise RefreshError(f"events[{index}] must be an object")
        if item.get("status", "open") not in OPEN_EVENT_STATUSES:
            continue
        due = parse_due(item.get("dueAt"), timezone, f"events[{index}].dueAt")
        if due is not None and due <= boundary:
            result.append((due, item))
    return sorted(result, key=lambda pair: pair[0])


def active_process_count(jobs: list[dict[str, Any]], events: list[dict[str, Any]]) -> int:
    active_job_ids = {
        str(item.get("id"))
        for item in jobs
        if isinstance(item, dict) and item.get("status") in ACTIVE_PROCESS_STATUSES and item.get("id")
    }
    count = len(active_job_ids)
    for event in events:
        if not isinstance(event, dict):
            continue
        if (
            event.get("status", "open") in OPEN_EVENT_STATUSES
            and event.get("type") in {"oa", "interview"}
            and str(event.get("jobId", "")) not in active_job_ids
        ):
            count += 1
    return count


def action(
    category: str,
    priority: int,
    title: str,
    detail: str,
    estimated_minutes: int,
    source: str,
    due_at: datetime | None = None,
) -> dict[str, Any]:
    return {
        "category": category,
        "priority": priority,
        "title": title,
        "detail": detail,
        "estimatedMinutes": max(0, int(estimated_minutes)),
        "source": source,
        "dueAt": due_at.isoformat() if due_at else None,
        "requiresConfirmation": True,
    }


def private_job_label(item: dict[str, Any]) -> str:
    company = str(item.get("company") or "未命名公司")
    title = str(item.get("title") or "未命名岗位")
    return f"{company} · {title}"


def collect_plan_market_actions(
    plan: dict[str, Any],
    now: datetime,
    timezone: ZoneInfo,
) -> list[dict[str, Any]]:
    raw_items = plan.get("urgentItems", plan.get("timeSensitiveItems", []))
    if raw_items is None:
        return []
    if not isinstance(raw_items, list):
        raise RefreshError("plan urgentItems must be an array")
    actions: list[dict[str, Any]] = []
    for index, item in enumerate(raw_items):
        if not isinstance(item, dict):
            raise RefreshError(f"urgentItems[{index}] must be an object")
        status = item.get("status")
        if status not in {"live", "verify_now"}:
            continue
        label = str(item.get("label") or item.get("title") or "time-sensitive role")
        source_url = item.get("sourceUrl")
        verified_at = item.get("verifiedAt")
        anticipated_close = item.get("anticipatedClose")
        reverify_by = item.get("reverifyBy")
        complete = all(isinstance(value, str) and value for value in (source_url, verified_at, anticipated_close, reverify_by))
        if not complete:
            actions.append(
                action(
                    "market-reverify",
                    4,
                    f"重新核验岗位线索：{label}",
                    "缺少 sourceUrl / verifiedAt / anticipatedClose / reverifyBy；核验前不得按 live 处理。",
                    15,
                    "weekly-plan.json",
                )
            )
            continue
        source_timezone = timezone
        if item.get("sourceTimezone"):
            try:
                source_timezone = ZoneInfo(str(item["sourceTimezone"]))
            except Exception as exc:
                raise RefreshError(
                    f"urgentItems[{index}].sourceTimezone is unknown: {item['sourceTimezone']}"
                ) from exc
        close_value = item.get("staleAfter") or anticipated_close
        close_at = parse_due(close_value, source_timezone, f"urgentItems[{index}].staleAfter")
        reverify_at = parse_due(reverify_by, source_timezone, f"urgentItems[{index}].reverifyBy")
        if close_at is None or reverify_at is None:
            continue
        if now > close_at:
            # A past anticipated close is no longer an urgent instruction. The
            # source can re-enter the queue only after the plan is reverified.
            continue
        requires_reverification = status == "verify_now" or item.get("requiresReverification") is True
        if requires_reverification or now > reverify_at:
            priority = 1 if close_at <= now + timedelta(hours=72) else 4
            actions.append(
                action(
                    "market-reverify",
                    priority,
                    f"先核验，不直接申请：{label}",
                    f"重新打开主来源确认状态、资格与材料；核验前不得当作 live。来源：{source_url}",
                    15,
                    "weekly-plan.json",
                    close_at if priority == 1 else None,
                )
            )
        elif close_at <= now + timedelta(hours=72):
            actions.append(
                action(
                    "verified-market-deadline",
                    1,
                    f"人工确认临近窗口：{label}",
                    f"先重新打开主来源，再人工决定是否申请。来源：{source_url}",
                    30,
                    "weekly-plan.json",
                    close_at,
                )
            )
    return actions


def structured_week_task(week: dict[str, Any], category: str) -> dict[str, Any] | None:
    tasks = week.get("tasks", [])
    if tasks is None:
        return None
    if not isinstance(tasks, list):
        raise RefreshError(f"{week.get('id', 'week')}.tasks must be an array")
    for index, item in enumerate(tasks):
        if not isinstance(item, dict):
            raise RefreshError(f"{week.get('id', 'week')}.tasks[{index}] must be an object")
        if item.get("category") == category:
            return item
    return None


def collect_queue(
    plan: dict[str, Any],
    week: dict[str, Any],
    config: dict[str, Any],
    jobs: list[dict[str, Any]],
    contacts: list[dict[str, Any]],
    events: list[dict[str, Any]],
    week_activity: list[dict[str, Any]],
    now: datetime,
    timezone: ZoneInfo,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    actions: list[dict[str, Any]] = []
    urgent_events = open_events_within(events, now, timezone, 72)
    for due, item in urgent_events:
        event_type = str(item.get("type") or "event")
        title = str(item.get("title") or event_type)
        minutes = int(item.get("estimatedMinutes") or (90 if event_type == "oa" else 60 if event_type == "interview" else 20))
        actions.append(
            action(
                "event",
                1,
                f"{event_type.upper()}：{title}",
                f"到期：{due.isoformat()}。先核对正式通知，再人工处理。",
                minutes,
                "events.json",
                due,
            )
        )
    actions.extend(collect_plan_market_actions(plan, now, timezone))

    urgent_job_ids = {
        str(item.get("jobId"))
        for _, item in urgent_events
        if item.get("jobId") not in (None, "")
    }
    overdue_count = 0
    overdue_contact_action = False
    for index, item in enumerate(jobs):
        if not isinstance(item, dict):
            raise RefreshError(f"jobs[{index}] must be an object")
        if str(item.get("id", "")) in urgent_job_ids:
            continue
        due = parse_due(item.get("nextActionDue"), timezone, f"jobs[{index}].nextActionDue")
        next_action = item.get("nextAction")
        if due is not None and due.date() <= now.date() and next_action and item.get("status") not in {"rejected", "withdrawn", "closed"}:
            overdue_count += 1
            actions.append(
                action(
                    "overdue-next-action",
                    2,
                    f"逾期岗位动作：{private_job_label(item)}",
                    str(next_action),
                    20,
                    "jobs.json",
                    due,
                )
            )
    for index, item in enumerate(contacts):
        if not isinstance(item, dict):
            raise RefreshError(f"contacts[{index}] must be an object")
        due = parse_due(item.get("nextActionDue"), timezone, f"contacts[{index}].nextActionDue")
        next_action = item.get("nextAction")
        if due is not None and due.date() <= now.date() and next_action and item.get("status") not in {"closed", "do-not-contact"}:
            overdue_count += 1
            overdue_contact_action = True
            label = f"{item.get('name') or '未命名联系人'} · {item.get('company') or '未知机构'}"
            actions.append(
                action(
                    "overdue-next-action",
                    2,
                    f"逾期关系动作：{label}",
                    str(next_action),
                    15,
                    "contacts.json",
                    due,
                )
            )

    project_preempted = False
    for index, item in enumerate(events):
        if not isinstance(item, dict) or item.get("status", "open") not in OPEN_EVENT_STATUSES:
            continue
        due = parse_due(item.get("dueAt"), timezone, f"events[{index}].dueAt")
        if due is None:
            continue
        if item.get("type") == "oa" and due <= now + timedelta(hours=72):
            project_preempted = True
        if item.get("type") == "interview" and due <= now + timedelta(days=5):
            project_preempted = True
    artifact_task = structured_week_task(week, "artifact")
    artifact = str(
        (artifact_task or {}).get("title")
        or week.get("artifact")
        or "完成本周最小可审查产物"
    )
    artifact_detail = str(
        (artifact_task or {}).get("evidenceRequired")
        or week.get("gate")
        or "保留可审查证据"
    )
    weekly_artifact_estimate = int((artifact_task or {}).get("estimateMinutes") or 180)
    artifact_minutes = 20 if project_preempted else min(90, max(30, round(weekly_artifact_estimate / 3)))
    if project_preempted:
        artifact_detail = f"OA/面试触发抢占：新功能归零，只保留文档、修复或恢复步骤。原 gate：{artifact_detail}"
    actions.append(
        action(
            "weekly-artifact",
            3,
            f"本周产物：{artifact}",
            artifact_detail,
            artifact_minutes,
            "weekly-plan.json",
        )
    )

    active_processes = active_process_count(jobs, events)
    application_cap = max(0, int(config.get("weeklyApplicationCap", 4)))
    completed_applications = sum(1 for item in week_activity if item.get("type") == "application")
    max_verification_age = max(0, int(config.get("maxJobVerificationAgeDays", 7)))
    qualified_jobs: list[dict[str, Any]] = []
    for index, item in enumerate(jobs):
        if (
            not isinstance(item, dict)
            or item.get("status") not in PRE_APPLICATION_STATUSES
            or item.get("fit") != "qualified"
            or item.get("eligibility") != "verified"
            or item.get("acceptable") is not True
            or not isinstance(item.get("sourceUrl"), str)
            or not item.get("sourceUrl")
            or not item.get("lastVerifiedAt")
        ):
            continue
        verified_date = parse_iso_date(item["lastVerifiedAt"], f"jobs[{index}].lastVerifiedAt")
        verification_age = (now.date() - verified_date).days
        if verification_age < 0 or verification_age > max_verification_age:
            continue
        deadline = parse_due(item.get("deadline"), timezone, f"jobs[{index}].deadline")
        if deadline is not None and deadline < now:
            continue
        qualified_jobs.append(item)
    qualified_jobs.sort(
        key=lambda item: (
            int(item.get("priority", 999)),
            str(item.get("deadline") or "9999-12-31"),
            private_job_label(item),
        )
    )
    if qualified_jobs and completed_applications < application_cap:
        selected = qualified_jobs[0]
        actions.append(
            action(
                "market-application",
                4,
                f"人工复核并决定是否申请：{private_job_label(selected)}",
                str(selected.get("nextAction") or "核验资格、JD 证据映射和材料版本；确认后才提交。"),
                45,
                "jobs.json",
                parse_due(selected.get("deadline"), timezone, "selected job deadline"),
            )
        )
    else:
        reason = (
            "本周申请容量已用完；只处理状态和高价值新线索。"
            if completed_applications >= application_cap
            else "没有满足 qualified + verified eligibility + acceptable 的岗位；允许 0 申请。"
        )
        actions.append(
            action(
                "market-monitoring",
                4,
                "监测并人工核验岗位",
                reason,
                20,
                "jobs.json",
            )
        )

    practice_task = structured_week_task(week, "practice")
    practice = str(
        (practice_task or {}).get("title")
        or week.get("practice")
        or "完成一个与当前岗位缺口相关的 OA/面试训练块"
    )
    practice_minutes = min(
        60,
        max(30, round(int((practice_task or {}).get("estimateMinutes") or 60) / 2)),
    )
    actions.append(
        action(
            "oa-practice",
            5,
            "OA／面试训练",
            practice,
            practice_minutes,
            "weekly-plan.json",
        )
    )

    completed_connections = sum(1 for item in week_activity if item.get("type") == "connection")
    connection_cap = max(0, int(config.get("weeklyConnectionCap", 3)))
    if active_processes < 2 and completed_connections < connection_cap and not overdue_contact_action:
        upcoming: list[tuple[datetime, dict[str, Any]]] = []
        for index, item in enumerate(contacts):
            if not isinstance(item, dict) or item.get("status") in {"closed", "do-not-contact"}:
                continue
            if int(item.get("followUpCount") or 0) >= 2 and item.get("status") != "warm":
                continue
            due = parse_due(item.get("nextActionDue"), timezone, f"contacts[{index}].nextActionDue")
            if due is not None and now.date() < due.date() <= (now + timedelta(days=7)).date() and item.get("nextAction"):
                upcoming.append((due, item))
        upcoming.sort(key=lambda pair: pair[0])
        if upcoming:
            due, selected_contact = upcoming[0]
            label = f"{selected_contact.get('name') or '未命名联系人'} · {selected_contact.get('company') or '未知机构'}"
            actions.append(
                action(
                    "connection",
                    6,
                    f"准备关系动作：{label}",
                    f"{selected_contact.get('nextAction')}。先检查上下文与频率，人工确认后再发送。",
                    20,
                    "contacts.json",
                    due,
                )
            )
        elif config.get("allowColdOutreach") is True:
            actions.append(
                action(
                    "connection",
                    6,
                    "研究 1 位相关从业者并起草个性化消息",
                    "必须有明确相关性与具体问题；不批量发送，发送前人工确认。",
                    25,
                    "contacts.json",
                )
            )

    actions.sort(key=lambda item: (item["priority"], item["dueAt"] or "9999", item["title"]))
    capacity = max(0, int(config.get("dailyCapacityMinutes", 120)))
    scheduled_minutes = 0
    capacity_blocked = False
    for item in actions:
        estimate = item["estimatedMinutes"]
        if item["priority"] == 1:
            item["scheduledToday"] = True
            scheduled_minutes += estimate
        elif not capacity_blocked and scheduled_minutes + estimate <= capacity:
            item["scheduledToday"] = True
            scheduled_minutes += estimate
        else:
            item["scheduledToday"] = False
            capacity_blocked = True
    stats = {
        "overdueNextActions": overdue_count,
        "activeProcesses": active_processes,
        "qualifiedOpenJobs": len(qualified_jobs),
        "scheduledMinutes": scheduled_minutes,
    }
    return actions, stats


def markdown_escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_private_markdown(
    selected_date: date,
    generated_at: datetime,
    plan: dict[str, Any],
    week: dict[str, Any],
    range_note: str | None,
    verified_level: str | None,
    evidence_counts: dict[str, int],
    actions: list[dict[str, Any]],
    metrics: dict[str, int],
    capacity: int,
) -> str:
    gate_target = week.get("gateTarget")
    scheduled = [item for item in actions if item["scheduledToday"]]
    parked = [item for item in actions if not item["scheduledToday"]]
    lines = [
        f"# {selected_date.isoformat()} AI 实习行动队列",
        "",
        "> 这是人工确认队列。脚本没有投递、发送、预约或修改任何外部系统。",
        "",
        f"- 计划版本：`{markdown_escape(plan.get('planVersion', 'unknown'))}`",
        f"- 当前周：`{markdown_escape(week.get('id', 'unknown'))}` · {markdown_escape(week.get('headline', ''))}",
        f"- 已验证 readiness：`{verified_level or '未验证'}`",
        f"- 本周 gate target：`{gate_target or '无升级 gate'}`（目标不等于当前等级）",
        f"- 今日容量：{capacity} 分钟；已排入 {metrics['scheduledMinutes']} 分钟",
        f"- 生成时间：{generated_at.isoformat()}",
    ]
    if range_note:
        lines.append(f"- 日期提示：{range_note}")
    lines.extend(
        [
            "",
            "## 今日先做",
            "",
            "| 优先级 | 类别 | 动作 | 预计 | 来源 |",
            "|---:|---|---|---:|---|",
        ]
    )
    for item in scheduled:
        due = f"；到期 {item['dueAt']}" if item.get("dueAt") else ""
        lines.append(
            f"| {item['priority']} | {markdown_escape(item['category'])} | "
            f"**{markdown_escape(item['title'])}** — {markdown_escape(item['detail'])}{markdown_escape(due)} | "
            f"{item['estimatedMinutes']} 分钟 | `{markdown_escape(item['source'])}` |"
        )
    if not scheduled:
        lines.append("| — | — | 没有自动排入的动作；人工决定是否休息或复盘。 | 0 | — |")
    lines.extend(
        [
            "",
            "## 容量外候选",
            "",
        ]
    )
    if parked:
        for item in parked:
            lines.append(
                f"- P{item['priority']} `{markdown_escape(item['category'])}`："
                f"{markdown_escape(item['title'])}（{item['estimatedMinutes']} 分钟）"
            )
    else:
        lines.append("- 无。")
    lines.extend(
        [
            "",
            "## 本周公开上下文",
            "",
            f"- 阶段：{markdown_escape(week.get('phase', ''))}",
            f"- 产物：{markdown_escape(week.get('artifact', ''))}",
            f"- 市场：{markdown_escape(week.get('market', ''))}",
            f"- 训练：{markdown_escape(week.get('practice', ''))}",
            f"- Gate：{markdown_escape(week.get('gate', ''))}",
            "",
            "计划中的市场描述只提供上下文；没有结构化且未过期的主来源核验时，不得把它当作 live deadline。",
            "",
            "## 本地状态摘要",
            "",
            f"- 72 小时内开放事件：{metrics['openEventsWithin72h']}",
            f"- 逾期下一步：{metrics['overdueNextActions']}",
            f"- 合格开放岗位：{metrics['qualifiedOpenJobs']}",
            f"- 活跃 OA／面试流程：{metrics['activeProcesses']}",
            f"- 本周已记录活动：{metrics['activityThisWeek']}",
            f"- Readiness 证据数：{', '.join(f'{level}={evidence_counts[level]}' for level in READINESS_LEVELS)}",
            "",
            "## 完成后人工记账",
            "",
            "只在真实完成后更新本地 JSON：岗位状态写入 `jobs.json`，关系动作写入 `contacts.json`，"
            "截止事项写入 `events.json`，已完成时间与结果追加到 `activity.json`。不要把私人文件导入网站。",
            "",
        ]
    )
    return "\n".join(lines)


def sanitized_snapshot(
    selected_date: date,
    generated_at: datetime,
    plan: dict[str, Any],
    week: dict[str, Any],
    verified_level: str | None,
    evidence_counts: dict[str, int],
    jobs: list[dict[str, Any]],
    contacts: list[dict[str, Any]],
    actions: list[dict[str, Any]],
    metrics: dict[str, int],
    capacity: int,
) -> dict[str, Any]:
    scheduled_actions = [item for item in actions if item["scheduledToday"]]
    category_counts = Counter(item["category"] for item in scheduled_actions)
    categories = list(dict.fromkeys(item["category"] for item in scheduled_actions))
    public_summary = [
        {
            "category": category,
            "count": category_counts[category],
            "label": PUBLIC_CATEGORY_LABELS[category],
        }
        for category in categories
    ]
    submitted = sum(
        1 for item in jobs if isinstance(item, dict) and item.get("status") in SUBMITTED_STATUSES
    )
    active_contacts = sum(
        1
        for item in contacts
        if isinstance(item, dict) and item.get("status") not in {"closed", "do-not-contact"}
    )
    gate_target = week.get("gateTarget")
    return {
        "schemaVersion": "ai-internship-site-snapshot.v1",
        "generatedAt": generated_at.isoformat(),
        "planVersion": str(plan.get("planVersion") or "unknown"),
        "date": selected_date.isoformat(),
        "week": {
            "id": week.get("id"),
            "start": week.get("start"),
            "end": week.get("end"),
            "phase": week.get("phase"),
            "headline": week.get("headline"),
            "artifact": week.get("artifact"),
            "gate": week.get("gate"),
            "gateTarget": gate_target,
        },
        "readiness": {
            "verifiedLevel": verified_level,
            "gateTarget": gate_target,
            "verifiedEvidenceCounts": evidence_counts,
        },
        "metrics": {
            "jobRecords": len(jobs),
            "qualifiedOpenJobs": metrics["qualifiedOpenJobs"],
            "activeProcesses": metrics["activeProcesses"],
            "applicationsSubmitted": submitted,
            "contactsTracked": active_contacts,
            "openEventsWithin72h": metrics["openEventsWithin72h"],
            "overdueNextActions": metrics["overdueNextActions"],
            "activityThisWeek": metrics["activityThisWeek"],
        },
        "today": {
            "capacityMinutes": capacity,
            "queuedCount": len(scheduled_actions),
            "categories": categories,
            "publicSummary": public_summary,
        },
        "privacy": {
            "sanitized": True,
            "scope": "device-local-records",
            "omitted": [
                "localCompany",
                "localJobTitle",
                "contactName",
                "localRecordId",
                "sourceUrl",
                "privateNotes",
                "individualDeadline",
            ],
        },
    }


def refresh(root: Path, plan_path: Path, requested_date: date | None) -> tuple[Path, Path]:
    config = read_json(root / "config.json", dict)
    jobs = read_json(root / "jobs.json", list)
    contacts = read_json(root / "contacts.json", list)
    events = read_json(root / "events.json", list)
    activity = read_json(root / "activity.json", list)
    plan = read_json(plan_path, dict)

    timezone_name = str(config.get("timezone") or plan.get("timezone") or "Asia/Shanghai")
    try:
        timezone = ZoneInfo(timezone_name)
    except Exception as exc:
        raise RefreshError(f"unknown timezone: {timezone_name}") from exc
    if requested_date is None:
        now = datetime.now(timezone)
        selected_date = now.date()
    else:
        selected_date = requested_date
        now = datetime.combine(selected_date, time(9, 0), tzinfo=timezone)

    week, range_note = choose_week(plan, selected_date)
    start, end = week_bounds(week)
    weekly_activity = activity_in_week(activity, start, end)
    verified_level, evidence_counts = verified_readiness(config)
    actions, queue_stats = collect_queue(
        plan,
        week,
        config,
        jobs,
        contacts,
        events,
        weekly_activity,
        now,
        timezone,
    )
    urgent_events = open_events_within(events, now, timezone, 72)
    metrics = {
        **queue_stats,
        "openEventsWithin72h": len(urgent_events),
        "activityThisWeek": len(weekly_activity),
    }
    capacity = max(0, int(config.get("dailyCapacityMinutes", 120)))
    private_text = render_private_markdown(
        selected_date,
        now,
        plan,
        week,
        range_note,
        verified_level,
        evidence_counts,
        actions,
        metrics,
        capacity,
    )
    snapshot = sanitized_snapshot(
        selected_date,
        now,
        plan,
        week,
        verified_level,
        evidence_counts,
        jobs,
        contacts,
        actions,
        metrics,
        capacity,
    )
    daily_path = root / "daily" / f"{selected_date.isoformat()}.md"
    snapshot_path = root / "exports" / "site-snapshot.json"
    atomic_write_text(daily_path, private_text)
    atomic_write_json(snapshot_path, snapshot)
    return daily_path, snapshot_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a private daily AI internship queue and sanitized site snapshot."
    )
    parser.add_argument("--root", required=True, type=Path, help="Private local state directory")
    parser.add_argument("--init", action="store_true", help="Copy missing state templates and pinned plan")
    parser.add_argument("--date", help="Planning/replay date in YYYY-MM-DD")
    parser.add_argument("--plan", type=Path, help="Reviewed plan JSON; defaults to ROOT/weekly-plan.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        selected_date = parse_iso_date(args.date, "--date") if args.date else None
        if args.init:
            created = initialize_state(args.root)
            if created:
                print("Initialized: " + ", ".join(str(path) for path in created))
            else:
                print("Initialization skipped existing files; nothing overwritten.")
        plan_path = args.plan or (args.root / "weekly-plan.json")
        daily_path, snapshot_path = refresh(args.root, plan_path, selected_date)
    except RefreshError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"Private daily queue: {daily_path}")
    print(f"Sanitized site snapshot: {snapshot_path}")
    print("No application or message was sent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
