ENVS = NUMBAPRO_NVVM=/opt/cuda/nvvm/lib64/libnvvm.so \
	NUMBAPRO_LIBDEVICE=/opt/cuda/nvvm/libdevice

help:
	echo 'make even-odd-sort -> a experiment to make a even-odd sort'

even-odd-sort:
	env $(ENVS) python even-odd-sort.py
