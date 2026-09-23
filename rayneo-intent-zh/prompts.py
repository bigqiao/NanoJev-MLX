"""Versioned Chinese intent question templates for local, synthetic evaluation.

The baseline is byte-for-byte the prompt used by the backend and by model
training. Candidate profiles are experimental until separately wired into the
backend. Gold labels and case groups are never included in these templates.
"""

PROFILES = {
    "baseline": {
        "assist": {"type": "boolean", "instructions": "当前说话者是否正在明确求助，或提出需要眼镜助手立即提供简短事实或解释的问题？普通闲聊、自问自答、假设对话不算。"},
        "schedule": {"type": "boolean", "instructions": "当前发言是否包含说话者确实打算参与、尚未取消且值得确认保存的日程安排？假设、取消、他人的无关事项不算。"},
    },
    "boundary_compact": {
        "assist": {"type": "boolean", "instructions": "结合上下文，只判断当前说话者这句话：是否明确向助手求助，或直接提出此刻需要助手回答的事实或解释问题？普通闲聊的疑问、自问自答、转述、已作决定或单纯更正为否。"},
        "schedule": {"type": "boolean", "instructions": "结合上下文，只判断当前说话者这句话：本人未来已确定参加，或明确要求记下、提醒的安排，是否值得先确认再保存？假设、待定、取消、他人的安排和普通时间数字为否。"},
    },
    "action_contrast": {
        "assist": {"type": "boolean", "instructions": "判断当前发言是否要助手现在帮忙：明确求助、直接询问事实或解释为是；聊天中的随口疑问、已有结论、复述他人、假设或更正为否。上下文只帮助理解当前发言。"},
        "schedule": {"type": "boolean", "instructions": "判断当前发言是否值得生成待确认日程卡：本人已约定或报名且未取消，或明确请助手记下、提醒具体安排为是；可能性讨论、他人的事、取消或普通问时间为否。"},
    },
    "boolean_criteria": {
        "assist": {"type": "boolean", "instructions": "当前说话者是否正在请助手帮助或回答此刻的问题？", "criteria": {
            "true": "明确求助或直接要求助手回答事实、解释。",
            "false": "普通聊天疑问、自问自答、已作决定、转述、假设或更正。",
        }},
        "schedule": {"type": "boolean", "instructions": "当前发言是否值得先确认再保存为本人的日程？", "criteria": {
            "true": "本人未来已确定的安排，或明确要求记下、提醒且尚未取消。",
            "false": "假设、待定、取消、他人事项或无关的时间数字。",
        }},
    },
}

# The fourth candidate preserves both trained question prefixes and only adds
# one short boundary sentence. It is an experiment, not a production default.
PROFILES["baseline_plus_guard"] = {
    "assist": {"type": "boolean", "instructions": PROFILES["baseline"]["assist"]["instructions"] + "只判断当前一句；已作出的决定或单纯更正不算求助。"},
    "schedule": {"type": "boolean", "instructions": PROFILES["baseline"]["schedule"]["instructions"] + "只判断当前一句；没有确定参加意图的时间数字不算。"},
}
