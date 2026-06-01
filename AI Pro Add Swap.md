要在你的 Orange Pi AI Pro（或任何 Linux 系统）上添加 12GB 的 swap 空间并永久生效，请按以下步骤操作：

## 1. 检查当前 swap 状态
```bash
free -h
```
如果已经有一些 swap，记下当前大小。

## 2. 创建 12GB 的 swap 文件
推荐使用 `fallocate`（速度快）：
```bash
sudo fallocate -l 12G /swapfile
```
## 3. 设置正确的权限（重要）
```bash
sudo chmod 600 /swapfile
```

## 4. 将文件格式化为 swap 分区
```bash
sudo mkswap /swapfile
```

## 5. 启用 swap 文件
```bash
sudo swapon /swapfile
```

## 6. 验证 swap 已启用
```bash
free -h
```
应该能看到 `/swapfile` 已激活，总 swap 增加了 12GB。

## 7. 设置永久生效（重启后自动挂载）
编辑 `/etc/fstab` 文件：
```bash
sudo vim /etc/fstab
```
在文件末尾添加以下一行：
```
/swapfile none swap sw 0 0
```
保存并退出（nano 中按 `Ctrl+O`，回车，`Ctrl+X`）。

## 验证永久性
重启系统后，运行 `free -h` 和 `swapon --show` 确认 swap 自动挂载。

**注意事项**：
- 确保磁盘有足够空间（12GB）。
- 如果系统根分区空间紧张，可以选择在其他分区创建 swap 文件，但上述步骤默认放在 `/` 下。
- 如果已有其他 swap 分区/文件，新添加的 swap 会与其共存，总容量为两者之和。

完成后，你的系统将拥有额外的 12GB swap 内存，对运行大型模型或内存密集型任务有帮助。