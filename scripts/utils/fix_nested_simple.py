#!/usr/bin/env python3
"""
简单直接的嵌套目录修复脚本
"""
import os
import shutil

TARGET = "/data/opentad-exps/exps"
NESTED = "/data/opentad-exps/exps/opentad-exps"

def move_all_to_target(source, target):
    """将所有内容从 source 移动到 target，递归处理嵌套"""
    if not os.path.exists(source) or not os.path.isdir(source):
        return
    
    items = list(os.listdir(source))
    if not items:
        return
    
    print(f"处理目录: {source} ({len(items)} 项)")
    
    for item_name in items:
        src = os.path.join(source, item_name)
        dst = os.path.join(target, item_name)
        
        # 如果是嵌套的 opentad-exps 或 exps，递归处理其内容
        if item_name in ["opentad-exps", "exps"]:
            print(f"  递归处理嵌套目录: {item_name}")
            move_all_to_target(src, target)  # 递归到目标目录
            # 删除空目录
            try:
                if os.path.exists(src) and not os.listdir(src):
                    os.rmdir(src)
                    print(f"    ✓ 删除空目录: {item_name}")
            except:
                pass
        else:
            # 普通文件或目录
            try:
                if os.path.exists(dst):
                    if os.path.isdir(src) and os.path.isdir(dst):
                        print(f"  合并目录: {item_name}")
                        move_all_to_target(src, dst)  # 递归合并
                        # 删除空目录
                        try:
                            if not os.listdir(src):
                                os.rmdir(src)
                        except:
                            pass
                    else:
                        print(f"  替换文件: {item_name}")
                        os.remove(dst)
                        shutil.move(src, dst)
                else:
                    print(f"  移动: {item_name}")
                    shutil.move(src, dst)
            except Exception as e:
                print(f"  ✗ 错误 {item_name}: {e}")

print("=" * 60)
print("修复嵌套目录")
print("=" * 60)

if os.path.exists(NESTED):
    print(f"\n发现嵌套目录: {NESTED}")
    move_all_to_target(NESTED, TARGET)
    
    # 删除空的嵌套目录
    try:
        if os.path.exists(NESTED) and not os.listdir(NESTED):
            os.rmdir(NESTED)
            print("\n✓ 嵌套目录已删除")
        else:
            # 如果还有内容，强制清理
            print("\n⚠ 嵌套目录不为空，强制清理...")
            for item in os.listdir(NESTED):
                item_path = os.path.join(NESTED, item)
                try:
                    if os.path.isdir(item_path):
                        move_all_to_target(item_path, TARGET)
                        shutil.rmtree(item_path)
                    else:
                        dst = os.path.join(TARGET, item)
                        if os.path.exists(dst):
                            os.remove(dst)
                        shutil.move(item_path, dst)
                except:
                    pass
            try:
                os.rmdir(NESTED)
                print("✓ 嵌套目录已删除")
            except:
                print("⚠ 无法删除嵌套目录")
    except Exception as e:
        print(f"\n⚠ 错误: {e}")
else:
    print("\n✓ 没有嵌套目录")

# 修复符号链接
print("\n修复符号链接...")
exps_link = "/root/OpenTAD/exps"
if os.path.islink(exps_link):
    current = os.readlink(exps_link)
    if current != TARGET:
        os.remove(exps_link)
        os.symlink(TARGET, exps_link)
        print(f"✓ 已更新: {current} -> {TARGET}")
    else:
        print("✓ 符号链接正确")
elif os.path.exists(exps_link):
    if os.path.isdir(exps_link):
        move_all_to_target(exps_link, TARGET)
        try:
            os.rmdir(exps_link)
        except:
            pass
    else:
        os.remove(exps_link)
    os.symlink(TARGET, exps_link)
    print("✓ 符号链接已创建")
else:
    os.symlink(TARGET, exps_link)
    print("✓ 符号链接已创建")

# 验证
print("\n" + "=" * 60)
print("验证结果")
print("=" * 60)

# 检查是否还有嵌套
nested_count = 0
for root, dirs, files in os.walk("/data/opentad-exps"):
    if "opentad-exps" in dirs:
        nested_path = os.path.join(root, "opentad-exps")
        if nested_path != TARGET:
            nested_count += 1
            print(f"  ⚠ 发现嵌套: {nested_path}")

if nested_count == 0:
    print("  ✓ 没有嵌套目录了")
else:
    print(f"  ⚠ 仍有 {nested_count} 个嵌套目录")

# 显示最终结构
if os.path.exists(TARGET):
    items = os.listdir(TARGET)
    print(f"\n目标目录内容 ({len(items)} 项):")
    for item in items:
        item_path = os.path.join(TARGET, item)
        item_type = "目录" if os.path.isdir(item_path) else "文件"
        print(f"  - {item} ({item_type})")

print("\n" + "=" * 60)
print("完成！")
print("=" * 60)

