#!/usr/bin/env python3
"""
强制修复所有嵌套目录 - 彻底清理
"""
import os
import shutil

TARGET = "/data/opentad-exps/exps"
EXPS_LINK = "/root/OpenTAD/exps"

def collect_all_items(source_path, target_path, collected=None):
    """收集所有需要移动的文件和目录"""
    if collected is None:
        collected = []
    
    if not os.path.exists(source_path) or not os.path.isdir(source_path):
        return collected
    
    for item_name in os.listdir(source_path):
        src = os.path.join(source_path, item_name)
        dst = os.path.join(target_path, item_name)
        
        # 跳过嵌套的 opentad-exps 和 exps 目录，递归处理其内容
        if item_name in ["opentad-exps", "exps"]:
            # 递归收集嵌套目录中的内容
            collect_all_items(src, target_path, collected)
        else:
            collected.append((src, dst))
            # 如果是目录，也收集其内容
            if os.path.isdir(src):
                collect_all_items(src, dst, collected)
    
    return collected

def merge_all_nested(source_path, target_path, depth=0, max_depth=20):
    """递归合并所有嵌套内容"""
    if depth > max_depth:
        print(f"{'  ' * depth}⚠ 达到最大深度 {max_depth}，停止")
        return
    
    if not os.path.exists(source_path) or not os.path.isdir(source_path):
        return
    
    items = list(os.listdir(source_path))
    if not items:
        return
    
    indent = "  " * depth
    print(f"{indent}[深度 {depth}] 处理: {source_path} ({len(items)} 项)")
    
    # 先处理嵌套的 opentad-exps 和 exps
    nested_items = [item for item in items if item in ["opentad-exps", "exps"]]
    other_items = [item for item in items if item not in ["opentad-exps", "exps"]]
    
    # 处理嵌套目录
    for item_name in nested_items:
        src = os.path.join(source_path, item_name)
        print(f"{indent}  → 递归处理嵌套目录: {item_name}")
        merge_all_nested(src, target_path, depth + 1, max_depth)
        # 尝试删除空目录
        try:
            if os.path.exists(src) and not os.listdir(src):
                os.rmdir(src)
                print(f"{indent}    ✓ 删除空目录: {item_name}")
        except:
            pass
    
    # 处理其他项目
    for item_name in other_items:
        src = os.path.join(source_path, item_name)
        dst = os.path.join(target_path, item_name)
        
        try:
            if os.path.exists(dst):
                if os.path.isdir(src) and os.path.isdir(dst):
                    print(f"{indent}  合并目录: {item_name}")
                    merge_all_nested(src, dst, depth + 1, max_depth)
                    # 尝试删除空目录
                    try:
                        if not os.listdir(src):
                            os.rmdir(src)
                    except:
                        pass
                else:
                    print(f"{indent}  替换文件: {item_name}")
                    os.remove(dst)
                    shutil.move(src, dst)
            else:
                print(f"{indent}  移动: {item_name}")
                shutil.move(src, dst)
        except Exception as e:
            print(f"{indent}  ✗ 错误 {item_name}: {e}")

def remove_all_nested_dirs(root_path, target, depth=0, max_depth=20):
    """删除所有嵌套的 opentad-exps 目录"""
    if depth > max_depth:
        return
    
    if not os.path.isdir(root_path):
        return
    
    items = list(os.listdir(root_path))
    for item_name in items:
        item_path = os.path.join(root_path, item_name)
        
        if item_name == "opentad-exps" and os.path.isdir(item_path):
            if item_path != target:
                # 递归处理
                remove_all_nested_dirs(item_path, target, depth + 1, max_depth)
                # 尝试删除
                try:
                    if not os.listdir(item_path):
                        os.rmdir(item_path)
                        print(f"  ✓ 删除空目录: {item_path}")
                except:
                    pass
        
        if os.path.isdir(item_path):
            remove_all_nested_dirs(item_path, target, depth + 1, max_depth)

print("=" * 60)
print("强制修复所有嵌套目录")
print("=" * 60)

# 确保目标目录存在
os.makedirs(TARGET, exist_ok=True)

# 1. 合并所有嵌套内容
print("\n[1/4] 合并所有嵌套内容...")
nested = os.path.join(TARGET, "opentad-exps")
if os.path.exists(nested):
    print(f"发现嵌套目录: {nested}")
    merge_all_nested(nested, TARGET, max_depth=20)
    
    # 尝试删除空的嵌套目录
    try:
        if os.path.exists(nested) and not os.listdir(nested):
            os.rmdir(nested)
            print("  ✓ 嵌套目录已删除")
        elif os.path.exists(nested):
            remaining = os.listdir(nested)
            print(f"  ⚠ 嵌套目录不为空，剩余: {remaining}")
            # 强制删除剩余内容
            for item in remaining:
                item_path = os.path.join(nested, item)
                try:
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                    else:
                        os.remove(item_path)
                except:
                    pass
            try:
                os.rmdir(nested)
                print("  ✓ 强制清理后删除嵌套目录")
            except:
                print("  ⚠ 无法删除嵌套目录")
    except Exception as e:
        print(f"  ⚠ 处理嵌套目录时出错: {e}")
else:
    print("  ✓ 没有嵌套目录")

# 2. 再次清理所有剩余的嵌套目录
print("\n[2/4] 清理所有剩余的嵌套目录...")
remove_all_nested_dirs("/data/opentad-exps", TARGET, max_depth=20)

# 3. 修复符号链接
print("\n[3/4] 修复符号链接...")
if os.path.islink(EXPS_LINK):
    current = os.readlink(EXPS_LINK)
    if current != TARGET:
        os.remove(EXPS_LINK)
        os.symlink(TARGET, EXPS_LINK)
        print(f"  ✓ 已更新: {current} -> {TARGET}")
    else:
        print("  ✓ 符号链接正确")
elif os.path.exists(EXPS_LINK):
    if os.path.isdir(EXPS_LINK):
        merge_all_nested(EXPS_LINK, TARGET, max_depth=20)
        try:
            os.rmdir(EXPS_LINK)
        except:
            pass
    else:
        os.remove(EXPS_LINK)
    os.symlink(TARGET, EXPS_LINK)
    print("  ✓ 符号链接已创建")
else:
    os.symlink(TARGET, EXPS_LINK)
    print("  ✓ 符号链接已创建")

# 4. 最终验证和清理
print("\n[4/4] 最终验证和清理...")

# 再次查找所有嵌套目录
def find_all_nested(path, target, depth=0, max_depth=20):
    if depth > max_depth:
        return []
    nested = []
    if os.path.isdir(path):
        for item in os.listdir(path):
            if item == "opentad-exps":
                item_path = os.path.join(path, item)
                if item_path != target:
                    nested.append(item_path)
            item_path = os.path.join(path, item)
            if os.path.isdir(item_path):
                nested.extend(find_all_nested(item_path, target, depth + 1, max_depth))
    return nested

remaining = find_all_nested("/data/opentad-exps", TARGET, max_depth=20)
if remaining:
    print(f"⚠ 仍有 {len(remaining)} 个嵌套目录，强制删除...")
    for n in remaining:
        print(f"  删除: {n}")
        try:
            if os.path.isdir(n):
                # 先合并内容
                merge_all_nested(n, TARGET, max_depth=20)
                # 再删除
                if not os.listdir(n):
                    os.rmdir(n)
                else:
                    shutil.rmtree(n)
        except Exception as e:
            print(f"    ✗ 错误: {e}")
else:
    print("✓ 没有嵌套目录了")

# 显示最终结构
print("\n" + "=" * 60)
print("最终结构")
print("=" * 60)

if os.path.exists(TARGET):
    items = os.listdir(TARGET)
    print(f"目标目录内容 ({len(items)} 项):")
    for item in items:
        item_path = os.path.join(TARGET, item)
        item_type = "目录" if os.path.isdir(item_path) else "文件"
        size = ""
        if os.path.isdir(item_path):
            try:
                count = len(os.listdir(item_path))
                size = f" ({count} 项)"
            except:
                pass
        print(f"  - {item} ({item_type}){size}")

print("\n" + "=" * 60)
print("修复完成！")
print("=" * 60)

