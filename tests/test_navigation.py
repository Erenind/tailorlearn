from tailorlearn.home_ui import NAVIGATION_CARDS
from tailorlearn.ui import NAV_GROUPS, NAV_ORDER, default_page


def test_sidebar_defaults_to_outline_and_keeps_management_pages_together() -> None:
    assert default_page() == "导航首页"
    assert NAV_GROUPS["开始使用"] == ("导航首页",)
    assert NAV_GROUPS["材料与管理"] == ("材料与数据", "教师工作台")
    assert NAV_ORDER.index("教师工作台") == NAV_ORDER.index("材料与数据") + 1


def test_navigation_home_exposes_all_primary_workflows() -> None:
    labels = [card["page"] for card in NAVIGATION_CARDS]

    assert labels == [
        "学习内容梳理",
        "材料与数据",
        "教师工作台",
        "学生分析",
        "班级分析",
        "教学智能体",
    ]
