from __future__ import annotations

TARGET_A4C_LABELS = (
    "DAO",
    "LA",
    "RA",
    "LV",
    "RV",
    "VS",
    "IS",
    "SP",
    "RB",
    "LVW",
    "RVW",
    "LL",
    "RL",
)

PUBLIC_SEGMENT_LABELS = (
    "SPINE",
    "RV",
    "LV",
    "RA",
    "LA",
    "AO",
    "PA",
    "SVC",
    "THREE_VESSELS",
    "AO_LVOT_CV",
    "AO_LVOT_OV",
)

CANONICAL_LABEL_METADATA = {
    "DAO": {
        "display_name": "descending aorta",
        "zh": "降主动脉",
        "prompts": [
            "segment the descending aorta in this fetal four-chamber ultrasound image",
            "highlight the fetal descending aorta",
            "标出胎儿四腔心图像中的降主动脉",
        ],
    },
    "LA": {
        "display_name": "left atrium",
        "zh": "左心房",
        "prompts": [
            "segment the left atrium in the fetal heart ultrasound",
            "highlight the fetal left atrium",
            "标出胎儿左心房",
        ],
    },
    "RA": {
        "display_name": "right atrium",
        "zh": "右心房",
        "prompts": [
            "segment the right atrium in the fetal heart ultrasound",
            "highlight the fetal right atrium",
            "标出胎儿右心房",
        ],
    },
    "LV": {
        "display_name": "left ventricle",
        "zh": "左心室",
        "prompts": [
            "segment the left ventricle in the fetal heart ultrasound",
            "highlight the fetal left ventricle",
            "标出胎儿左心室",
        ],
    },
    "RV": {
        "display_name": "right ventricle",
        "zh": "右心室",
        "prompts": [
            "segment the right ventricle in the fetal heart ultrasound",
            "highlight the fetal right ventricle",
            "标出胎儿右心室",
        ],
    },
    "VS": {
        "display_name": "ventricular septum",
        "zh": "室间隔",
        "prompts": [
            "segment the ventricular septum in the fetal heart ultrasound",
            "highlight the fetal ventricular septum",
            "标出胎儿室间隔",
        ],
    },
    "IS": {
        "display_name": "interatrial septum",
        "zh": "房间隔",
        "prompts": [
            "segment the interatrial septum in the fetal heart ultrasound",
            "highlight the fetal interatrial septum",
            "标出胎儿房间隔",
        ],
    },
    "SP": {
        "display_name": "spine",
        "zh": "脊柱",
        "prompts": [
            "segment the fetal spine in this ultrasound image",
            "highlight the fetal spine",
            "标出胎儿脊柱",
        ],
    },
    "RB": {
        "display_name": "rib",
        "zh": "肋骨",
        "prompts": [
            "segment the fetal rib region in this ultrasound image",
            "highlight the fetal rib",
            "标出胎儿肋骨",
        ],
    },
    "LVW": {
        "display_name": "left ventricular wall",
        "zh": "左心室壁",
        "prompts": [
            "segment the left ventricular wall in the fetal heart ultrasound",
            "highlight the left ventricular wall",
            "标出胎儿左心室壁",
        ],
    },
    "RVW": {
        "display_name": "right ventricular wall",
        "zh": "右心室壁",
        "prompts": [
            "segment the right ventricular wall in the fetal heart ultrasound",
            "highlight the right ventricular wall",
            "标出胎儿右心室壁",
        ],
    },
    "LL": {
        "display_name": "left lung",
        "zh": "左肺",
        "prompts": [
            "segment the left lung in this fetal ultrasound image",
            "highlight the fetal left lung",
            "标出胎儿左肺",
        ],
    },
    "RL": {
        "display_name": "right lung",
        "zh": "右肺",
        "prompts": [
            "segment the right lung in this fetal ultrasound image",
            "highlight the fetal right lung",
            "标出胎儿右肺",
        ],
    },
    "SPINE": {
        "display_name": "spine",
        "zh": "脊柱",
        "prompts": [
            "segment the fetal spine in this ultrasound image",
            "highlight the fetal spine landmark",
            "标出胎儿脊柱",
        ],
    },
    "AO": {
        "display_name": "aorta",
        "zh": "主动脉",
        "prompts": [
            "segment the aorta in this fetal echocardiography image",
            "highlight the fetal aorta",
            "标出胎儿主动脉",
        ],
    },
    "PA": {
        "display_name": "pulmonary artery",
        "zh": "肺动脉",
        "prompts": [
            "segment the pulmonary artery in this fetal echocardiography image",
            "highlight the fetal pulmonary artery",
            "标出胎儿肺动脉",
        ],
    },
    "SVC": {
        "display_name": "superior vena cava",
        "zh": "上腔静脉",
        "prompts": [
            "segment the superior vena cava in this fetal echocardiography image",
            "highlight the fetal superior vena cava",
            "标出胎儿上腔静脉",
        ],
    },
    "THREE_VESSELS": {
        "display_name": "three-vessel view region",
        "zh": "三血管区域",
        "prompts": [
            "segment the three-vessel view region in this fetal ultrasound image",
            "highlight the fetal three-vessel region",
            "标出胎儿三血管区域",
        ],
    },
    "AO_LVOT_CV": {
        "display_name": "aorta left-ventricular outflow tract close view",
        "zh": "主动脉左室流出道近景",
        "prompts": [
            "segment the aorta and left ventricular outflow tract close-view region",
            "highlight the fetal aorta lvot close view",
            "标出主动脉与左室流出道近景区域",
        ],
    },
    "AO_LVOT_OV": {
        "display_name": "aorta left-ventricular outflow tract overview",
        "zh": "主动脉左室流出道概览",
        "prompts": [
            "segment the aorta and left ventricular outflow tract overview region",
            "highlight the fetal aorta lvot overview",
            "标出主动脉与左室流出道概览区域",
        ],
    },
}

PUBLIC_SECOND_TRIMESTER_LABEL_MAP = {
    "S": "SPINE",
    "VD": "RV",
    "VS": "LV",
    "AD": "RA",
    "AS": "LA",
    "Ao": "AO",
    "PA": "PA",
    "SVC": "SVC",
    "3VV": "THREE_VESSELS",
    "AoLVOTCV": "AO_LVOT_CV",
    "AoLVOTOV": "AO_LVOT_OV",
}
