IMAGE := sqrtspace-demo

.PHONY: build shell demo-barely demo-spill satcom-demo

build:
	docker build -t $(IMAGE) .

shell:
	docker run --rm -it \
		--memory 512m --memory-swap 2g \
		--cpus 1 \
		-e LLC_BYTES=33554432 \
		-v $(PWD):/app:ro $(IMAGE) bash

demo-barely:
	docker run --rm \
		--memory 512m --memory-swap 2g \
		--cpus 1 \
		-e LLC_BYTES=33554432 \
		-v $(PWD):/app $(IMAGE) bash -lc "sqrtspace bench logs --n 10000000 --win 32768 --hop 4096 --csv logs_barely.csv && python bench/plot_bench.py logs_barely.csv"

demo-spill:
	docker run --rm \
		--memory 384m --memory-swap 4g \
		--cpus 1 \
		-e LLC_BYTES=33554432 \
		-v $(PWD):/app $(IMAGE) bash -lc "sqrtspace bench logs --extreme --n 20000000 --win 65536 --hop 4096 --csv extreme_logs.csv && python bench/plot_memory.py extreme_logs.csv"

demo-spill-lite:
	docker run --rm \
		--memory 384m --memory-swap 4g \
		--cpus 1 \
		-e LLC_BYTES=33554432 \
		-v $(PWD):/app $(IMAGE) bash -lc "sqrtspace bench logs --extreme --n 12000000 --win 65536 --hop 4096 --csv extreme_logs_lite.csv && python bench/plot_memory.py extreme_logs_lite.csv"

satcom-demo:
	docker run --rm \
		--memory 512m --memory-swap 512m \
		--cpus 1 \
		-e LLC_BYTES=33554432 \
		-v $(PWD):/app $(IMAGE) bash -lc "sqrtspace bench satcom --deep --csv satcom_deep.csv && python bench/plot_bench.py satcom_deep.csv"

demo-spill-auto:
	docker run --rm \
		--memory 384m --memory-swap 8g \
		--cpus 1 \
		-e LLC_BYTES=33554432 \
		-v $(PWD):/app $(IMAGE) bash -lc 'set -e; for N in 12000000 10000000 8000000 6000000 4000000; do \
		  echo "Trying N=$$N"; \
		  if sqrtspace bench logs --extreme --n $$N --win 65536 --hop 4096 --csv extreme_logs_auto.csv; then \
		    python bench/plot_memory.py extreme_logs_auto.csv; echo "Succeeded with N=$$N"; exit 0; \
		  else \
		    echo "Failed with N=$$N, decreasing..."; \
		  fi; \
		done; exit 1'


