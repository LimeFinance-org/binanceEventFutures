PY ?= .venv/bin/python

install:        ## 创建虚拟环境并安装依赖（含 CUDA 版 PyTorch）
	bash deploy/install.sh
test:           ## 运行全部自动测试（离线，约 2 分钟）
	$(PY) -m pytest tests -q
data:           ## 下载/增量更新历史数据
	$(PY) -m becai.cli data-sync
train:          ## 启动全自动训练（可用 RUN=run_xxx 断点恢复）
	$(PY) -m becai.cli train $(if $(RUN),--run-id $(RUN),)
live:           ## 启动每分钟实时预测服务
	$(PY) -m becai.cli live
web:            ## 启动中文控制台
	$(PY) -m becai.cli web
replay:         ## 用历史数据回放实时流程
	$(PY) -m becai.cli replay --minutes 120
.PHONY: install test data train live web replay
