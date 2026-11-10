# systemd 部署

服务文件是模板单元（`becai-web@.service`），`@` 后面填运行服务的系统用户名。

```bash
sudo mkdir -p /opt/becai && sudo chown $USER /opt/becai
cp -r . /opt/becai && cd /opt/becai && bash deploy/install.sh
sudo cp deploy/systemd/becai-*@.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now becai-web@$USER becai-live@$USER becai-scheduler@$USER
journalctl -u becai-live@$USER -f      # 查看实时预测日志
```

三个服务相互独立：
- `becai-live` 只用 CPU（`CUDA_VISIBLE_DEVICES=` 为空）并绑定专用核心，训练再忙也不影响每分钟预测；
- 训练任务由控制台或调度器以子进程启动，自动避开实时服务的核心并降低优先级；
- `becai-scheduler` 负责每日数据更新、定期再训练、训练异常中断后的自动恢复。
