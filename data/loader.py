import json
import os
import random
import re
import uuid
from datetime import datetime, timedelta

from sklearn.model_selection import train_test_split

from config import CATEGORIES, SEVERITIES, DATA_PATH, TEST_SIZE

CATEGORY_WEIGHTS = [30, 20, 15, 12, 10, 8, 5]
SEVERITY_WEIGHTS = [5, 15, 40, 25, 15]

TEMPLATES = {
    "功能": [
        "用户点击{action}按钮后页面无响应，预期应{expected}，实际没有任何反馈",
        "在{module}模块中，执行{action}操作时系统报错：{error}，导致功能无法正常使用",
        "{module}功能的{action}逻辑存在异常，当输入{input}时返回结果不正确",
        "系统在{condition}条件下，{module}模块的{action}功能失效，影响用户正常操作",
        "用户无法完成{action}操作，{module}页面加载后功能按钮均为禁用状态",
    ],
    "性能": [
        "{module}模块在数据量超过{threshold}条时，查询响应时间超过{time}秒，严重影响用户体验",
        "系统在{condition}情况下，页面加载耗时{time}秒以上，CPU占用率达到{cpu}%",
        "并发用户数达到{users}时，{module}接口响应时间急剧增加，P99延迟超过{time}毫秒",
        "{module}功能在大数据量场景下存在内存泄漏，每次操作增加约{leak}MB内存占用",
        "系统启动{module}模块后，内存占用持续增长，{time}分钟内从{mem_start}MB升至{mem_end}MB",
    ],
    "UI": [
        "{module}页面在{browser}浏览器下布局错乱，{element}元素重叠显示",
        "在{resolution}分辨率下，{module}页面的{element}组件未正确适配，部分内容被截断",
        "{module}模块的{element}文字颜色与背景色对比度不足，无法清晰辨认内容",
        "切换到{language}语言后，{module}页面的{element}区域出现文字截断和排版错位",
        "{module}页面在移动端显示时，{element}按钮尺寸过小且间距不合理，难以点击",
    ],
    "接口": [
        "调用{module}的{api}接口返回500错误，错误信息：{error}，导致相关业务流程中断",
        "{api}接口在{condition}参数组合下返回数据格式与文档不一致，缺少{field}字段",
        "{module}模块的{api}接口响应超时，平均耗时{time}秒，远超服务等级协议要求",
        "并发请求{api}接口时出现数据竞争，部分请求返回了其他用户的数据",
        "{api}接口未对{param}参数进行校验，传入非法值时导致服务端异常崩溃",
    ],
    "数据": [
        "{module}模块导出的数据存在{error_type}错误，部分记录的{field}字段值丢失或重复",
        "数据同步时{module}表的{field}字段与源系统不一致，差异记录数达到{count}条",
        "{module}模块在{condition}场景下，数据写入存在延迟，最长延迟{time}分钟",
        "执行{action}操作后，{module}中的关联数据未正确更新，导致数据不一致",
        "{module}报表统计结果与原始数据对不上，{metric}指标偏差达到{deviation}%",
    ],
    "安全": [
        "{module}模块存在{vuln_type}漏洞，攻击者可通过{attack}方式获取未授权数据访问权限",
        "系统{api}接口未进行身份认证，任意用户可直接访问敏感{data_type}数据",
        "{module}页面的用户输入未进行过滤和转义，存在跨站脚本攻击风险",
        "数据库查询未使用参数化方式，{module}模块存在SQL注入安全隐患",
        "用户密码在{module}模块中以明文形式传输和存储，存在严重安全风险",
    ],
    "兼容性": [
        "{module}在{os}系统上运行时{element}功能异常，与Windows环境表现不一致",
        "使用{browser}浏览器访问{module}时，{element}组件渲染异常，页面无法正常交互",
        "{module}模块在{device}设备上{action}操作无效，点击后无任何响应",
        "系统在{os}平台下，{module}的{element}显示字体与预期不符，部分字符显示为方块",
        "{module}功能与第三方{third_party}组件存在兼容性问题，集成后导致系统不稳定",
    ],
}

_FILL_MAP = {
    "action": ["提交", "保存", "删除", "搜索", "导出", "导入", "审批", "登录", "注册", "查询", "更新", "下载"],
    "expected": ["跳转到结果页", "显示成功提示", "返回正确数据", "更新列表", "弹出确认对话框"],
    "module": ["用户管理", "订单处理", "库存管理", "报表统计", "权限控制", "消息通知", "数据导入", "审批流程", "支付结算", "日志查询"],
    "error": ["空指针异常", "超时未响应", "数据库连接失败", "权限不足", "参数校验失败"],
    "input": ["特殊字符", "超长文本", "空值", "负数", "非法格式"],
    "condition": ["高并发", "弱网环境", "数据量较大", "多用户同时操作", "长时间运行"],
    "threshold": ["10000", "50000", "100000", "5000"],
    "time": ["5", "10", "30", "60", "120"],
    "cpu": ["80", "90", "95", "85"],
    "users": ["100", "500", "1000", "2000"],
    "leak": ["2", "5", "10", "8"],
    "mem_start": ["200", "500", "1024"],
    "mem_end": ["800", "1500", "4096"],
    "browser": ["Safari", "Firefox", "Edge", "IE11", "Chrome"],
    "resolution": ["1920x1080", "1366x768", "2560x1440", "1280x720"],
    "element": ["表格", "表单", "导航栏", "弹窗", "下拉框", "按钮", "标签页", "日期选择器"],
    "language": ["英文", "日文", "韩文", "繁体中文"],
    "api": ["/api/users", "/api/orders", "/api/products", "/api/reports", "/api/auth"],
    "field": ["id", "name", "status", "amount", "createTime", "updateTime"],
    "param": ["id", "page", "sort", "filter", "keyword"],
    "error_type": ["数据丢失", "字段缺失", "格式错误", "编码异常"],
    "count": ["50", "200", "1000", "500"],
    "metric": ["总数", "平均值", "最大值", "占比"],
    "deviation": ["5", "10", "20", "15"],
    "vuln_type": ["越权访问", "信息泄露", "命令注入", "文件包含"],
    "attack": ["构造恶意请求", "篡改参数", "会话劫持", "暴力破解"],
    "data_type": ["用户隐私", "财务", "系统配置", "业务核心"],
    "os": ["macOS", "Linux Ubuntu", "Linux CentOS", "国产麒麟系统"],
    "device": ["iPad", "Android平板", "Surface", "华为MatePad"],
    "third_party": ["ECharts", "TinyMCE", "Swagger", "ELK"],
}


def _fill_template(template):
    placeholders = re.findall(r'\{(\w+)\}', template)
    result = template
    for ph in placeholders:
        if ph in _FILL_MAP:
            result = result.replace('{' + ph + '}', random.choice(_FILL_MAP[ph]), 1)
    return result


def generate_sample_data(n=5000):
    records = []
    base_time = datetime(2025, 1, 1, 0, 0, 0)
    for i in range(n):
        category = random.choices(CATEGORIES, weights=CATEGORY_WEIGHTS, k=1)[0]
        severity = random.choices(SEVERITIES, weights=SEVERITY_WEIGHTS, k=1)[0]
        template = random.choice(TEMPLATES[category])
        description = _fill_template(template)
        ts = base_time + timedelta(
            days=random.randint(0, 365),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59),
            seconds=random.randint(0, 59),
        )
        records.append({
            "id": str(uuid.uuid4()),
            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "description": description,
            "category": category,
            "severity": severity,
            "confirmer": "",
        })
    return records


def load_data():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_data(data):
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    tmp_path = DATA_PATH + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, DATA_PATH)


def add_records(records):
    data = load_data() if os.path.exists(DATA_PATH) else []
    data.extend(records)
    save_data(data)
    return data


def split_data(data):
    categories = [r["category"] for r in data]
    test_size = min(TEST_SIZE, len(data) // 5)
    test_size = max(test_size, 1)
    train, test = train_test_split(
        data,
        test_size=test_size,
        stratify=categories,
        random_state=42,
    )
    return train, test


def init_data_if_needed():
    if not os.path.exists(DATA_PATH):
        data = generate_sample_data(5000)
        save_data(data)
