#!/usr/bin/env python3
"""
递归修复深层嵌套目录
"""
import os
import shutil

TARGET = "/data/opentad-exps/exps"
EXPS_LINK = "/root/OpenTAD/exps"

def merge_all_to_target(source_path, target_path, depth=0, max_depth=10):
    """递归合并所有内容到目标目录"""
    if depth > max_depth:
        print(f"{'  ' * depth}⚠ 达到最大深度，停止递归")
        return
    
    if not os.path.exists(source_path) or not os.path.isdir(source_path):
        return
    
    items = list(os.listdir(source_path))
    if not items:
        return
    
    indent = "  " * depth
    print(f"{indent}处理目录: {source_path} ({len(items)} 项)")
    
    for item_name in items:
        src = os.path.join(source_path, item_name)
        dst = os.path.join(target_path, item_name)
        
        if item_name == "opentad-exps":
            # 递归处理嵌套的 opentad-exps
            print(f"{indent}  → 发现嵌套 opentad-exps，递归处理...")
            merge_all_to_target(src, target_path, depth + 1, max_depth)
            # 尝试删除空目录
            try:
                if not os.listdir(src):
                    os.rmdir(src)
                    print(f"{indent}  ✓ 删除空目录: {src}")
            except:
                pass
        elif item_name == "exps":
            # 合并嵌套的 exps 目录内容
            print(f"{indent}  → 发现嵌套 exps，合并内容...")
            merge_all_to_target(src, target_path, depth + 1, max_depth)
            # 尝试删除空目录
            try:
                if not os.listdir(src):
                    os.rmdir(src)
                    print(f"{indent}  ✓ 删除空 exps 目录")
            except:
                pass
        else:
            # 普通文件或目录
            try:
                if os.path.exists(dst):
                    if os.path.isdir(src) and os.path.isdir(dst):
                        # 合并目录
                        print(f"{indent}  合并目录: {item_name}")
                        merge_all_to_target(src, dst, depth + 1, max_depth)
                        # 尝试删除空目录
                        try:
                            if not os.listdir(src):
                                os.rmdir(src)
                        except:
                            pass
                    else:
                        # 替换文件
                        print(f"{indent}  替换文件: {item_name}")
                        os.remove(dst)
                        shutil.move(src, dst)
                else:
                    # 直接移动
                    print(f"{indent}  移动: {item_name}")
                    shutil.move(src, dst)
            except Exception as e:
                print(f"{indent}  ✗ 错误处理 {item_name}: {e}")

def remove_all_nested(path, target, depth=0, max_depth=10):
    """删除所有剩余的嵌套 opentad-exps 目录"""
    if depth > max_depth:
        return
    
    if not os.path.isdir(path):
        return
    
    items = list(os.listdir(path))
    for item_name in items:
        item_path = os.path.join(path, item_name)
        if item_name == "opentad-exps" and os.path.isdir(item_path):
            if item_path != target:
                remove_all_nested(item_path, target, depth + 1, max_depth)
                try:
                    if not os.listdir(item_path):
                        os.rmdir(item_path)
                        print(f"  ✓ 删除空目录: {item_path}")
                except:
                    pass
        elif os.path.isdir(item_path):
            remove_all_nested(item_path, target, depth + 1, max_depth)

print("=" * 60)
print("递归修复深层嵌套目录")
print("=" * 60)

# 确保目标目录存在
os.makedirs(TARGET, exist_ok=True)

# 1. 合并所有嵌套内容
print("\n[1/3] 合并所有嵌套内容...")
nested = os.path.join(TARGET, "opentad-exps")
if os.path.exists(nested):
    merge_all_to_target(nested, TARGET)
    # 尝试删除空的嵌套目录
    try:
        if not os.listdir(nested):
            os.rmdir(nested)
            print("  ✓ 嵌套目录已删除")
        else:
            remaining = os.listdir(nested)
            print(f"  ⚠ 嵌套目录不为空，剩余: {remaining}")
    except Exception as e:
        print(f"  ⚠ 无法删除嵌套目录: {e}")
else:
    print("  ✓ 没有嵌套目录")

# 2. 清理所有剩余的嵌套目录
print("\n[2/3] 清理所有嵌套目录...")
remove_all_nested("/data/opentad-exps", TARGET)

# 3. 修复符号链接
print("\n[3/3] 修复符号链接...")
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
        merge_all_to_target(EXPS_LINK, TARGET)
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

# 验证
print("\n" + "=" * 60)
print("验证结果")
print("=" * 60)

# 检查嵌套
def find_all_nested(path, target, depth=0, max_depth=10):
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

remaining = find_all_nested("/data/opentad-exps", TARGET)
if remaining:
    print(f"⚠ 仍有 {len(remaining)} 个嵌套目录:")
    for n in remaining:
        print(f"  - {n}")
else:
    print("✓ 没有嵌套目录了")

# 显示最终结构
if os.path.exists(TARGET):
    items = os.listdir(TARGET)
    print(f"\n目标目录内容 ({len(items)} 项):")
    for item in items[:10]:
        item_path = os.path.join(TARGET, item)
        item_type = "目录" if os.path.isdir(item_path) else "文件"
        print(f"  - {item} ({item_type})")

print("\n" + "=" * 60)
print("修复完成！")
print("=" * 60)

