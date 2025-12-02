#!/usr/bin/env python3
"""
修复嵌套的 exps 目录结构
"""
import os
import shutil
import sys

# 修复路径问题
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

DATA_DIR = "/data/opentad-exps"
EXPS_LINK = "/root/OpenTAD/exps"
TARGET_DIR = "/data/opentad-exps/exps"

print("=" * 60)
print("修复嵌套的 exps 目录结构")
print("=" * 60)

# 1. 查找所有嵌套的目录
def find_nested_exps(path, depth=0, max_depth=10):
    """递归查找嵌套的 exps 目录"""
    if depth > max_depth:
        return []
    
    nested = []
    if os.path.isdir(path):
        for item in os.listdir(path):
            item_path = os.path.join(path, item)
            if item == "opentad-exps" and os.path.isdir(item_path):
                nested.append(item_path)
            elif os.path.isdir(item_path):
                nested.extend(find_nested_exps(item_path, depth + 1, max_depth))
    return nested

print("\n[1/4] 查找嵌套目录...")
nested_dirs = find_nested_exps(DATA_DIR)
print(f"找到 {len(nested_dirs)} 个嵌套目录:")
for d in nested_dirs:
    print(f"  - {d}")

# 2. 合并所有嵌套目录的内容到目标目录
print("\n[2/4] 合并嵌套目录内容...")
os.makedirs(TARGET_DIR, exist_ok=True)

for nested_dir in nested_dirs:
    if os.path.exists(nested_dir):
        print(f"  处理: {nested_dir}")
        # 移动所有内容（除了 opentad-exps 本身）
        for item in os.listdir(nested_dir):
            src = os.path.join(nested_dir, item)
            dst = os.path.join(TARGET_DIR, item)
            
            if item == "opentad-exps":
                # 如果是嵌套的 opentad-exps，递归处理
                if os.path.isdir(src):
                    for sub_item in os.listdir(src):
                        sub_src = os.path.join(src, sub_item)
                        sub_dst = os.path.join(TARGET_DIR, sub_item)
                        if os.path.exists(sub_dst):
                            if os.path.isdir(sub_src) and os.path.isdir(sub_dst):
                                # 合并目录
                                for sub_sub in os.listdir(sub_src):
                                    shutil.move(
                                        os.path.join(sub_src, sub_sub),
                                        os.path.join(sub_dst, sub_sub)
                                    )
                                os.rmdir(sub_src)
                            else:
                                os.remove(sub_dst)
                                shutil.move(sub_src, sub_dst)
                        else:
                            shutil.move(sub_src, sub_dst)
                    os.rmdir(src)
            else:
                if os.path.exists(dst):
                    if os.path.isdir(src) and os.path.isdir(dst):
                        # 合并目录内容
                        for sub_item in os.listdir(src):
                            shutil.move(
                                os.path.join(src, sub_item),
                                os.path.join(dst, sub_item)
                            )
                        os.rmdir(src)
                    else:
                        os.remove(dst)
                        shutil.move(src, dst)
                else:
                    shutil.move(src, dst)
        
        # 尝试删除空目录
        try:
            os.rmdir(nested_dir)
            print(f"    ✓ 已删除空目录")
        except:
            print(f"    ⚠ 目录不为空，保留")

# 3. 清理所有剩余的嵌套 opentad-exps 目录
print("\n[3/4] 清理剩余的嵌套目录...")
def remove_nested_opentad_exps(path, depth=0, max_depth=10):
    if depth > max_depth:
        return
    
    if os.path.isdir(path):
        for item in os.listdir(path):
            item_path = os.path.join(path, item)
            if item == "opentad-exps" and os.path.isdir(item_path):
                # 检查是否为空
                try:
                    if not os.listdir(item_path):
                        os.rmdir(item_path)
                        print(f"  ✓ 删除空目录: {item_path}")
                    else:
                        # 递归处理
                        remove_nested_opentad_exps(item_path, depth + 1, max_depth)
                        # 再次尝试删除
                        if not os.listdir(item_path):
                            os.rmdir(item_path)
                except:
                    pass
            elif os.path.isdir(item_path):
                remove_nested_opentad_exps(item_path, depth + 1, max_depth)

remove_nested_opentad_exps(DATA_DIR)

# 4. 修复符号链接
print("\n[4/4] 修复符号链接...")
if os.path.islink(EXPS_LINK):
    current_target = os.readlink(EXPS_LINK)
    if current_target != TARGET_DIR:
        print(f"  当前目标: {current_target}")
        print(f"  更新为: {TARGET_DIR}")
        os.remove(EXPS_LINK)
        os.symlink(TARGET_DIR, EXPS_LINK)
        print("  ✓ 符号链接已更新")
    else:
        print("  ✓ 符号链接正确")
elif os.path.exists(EXPS_LINK):
    print(f"  exps 是目录，转换为符号链接...")
    # 移动内容
    if os.path.isdir(EXPS_LINK):
        for item in os.listdir(EXPS_LINK):
            shutil.move(
                os.path.join(EXPS_LINK, item),
                os.path.join(TARGET_DIR, item)
            )
        os.rmdir(EXPS_LINK)
    else:
        os.remove(EXPS_LINK)
    os.symlink(TARGET_DIR, EXPS_LINK)
    print("  ✓ 符号链接已创建")
else:
    os.symlink(TARGET_DIR, EXPS_LINK)
    print("  ✓ 符号链接已创建")

# 验证
print("\n" + "=" * 60)
print("验证最终结构")
print("=" * 60)

if os.path.islink(EXPS_LINK):
    real_path = os.path.realpath(EXPS_LINK)
    print(f"符号链接: {EXPS_LINK} -> {real_path}")
    
    if os.path.exists(real_path):
        print(f"✓ 目录存在")
        items = os.listdir(real_path)
        print(f"目录内容 ({len(items)} 项):")
        for item in items[:10]:
            item_path = os.path.join(real_path, item)
            item_type = "目录" if os.path.isdir(item_path) else "文件"
            print(f"  - {item} ({item_type})")
    else:
        print(f"✗ 目录不存在")

# 检查是否还有嵌套
remaining_nested = find_nested_exps(DATA_DIR)
if remaining_nested:
    print(f"\n⚠ 仍有 {len(remaining_nested)} 个嵌套目录:")
    for d in remaining_nested:
        print(f"  - {d}")
else:
    print("\n✓ 没有嵌套目录了")

print("\n" + "=" * 60)
print("修复完成！")
print("=" * 60)

