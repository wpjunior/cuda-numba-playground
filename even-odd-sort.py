import numba
from numba import cuda
import numpy as np


@cuda.jit
def sort_kernel(array):
    """
    Função executada dentro da GPU via LLVM
    """
    # posição do grid
    pos = cuda.grid(1) * 2

    for _ in range(array.size/2):
        # utilizado para evitar buffer overflow
        # não podemos acessar uma posição do array invalida
        if pos > 0:
            current = array[pos]
            left = array[pos - 1]

            array[pos - 1] = min(left, current)
            array[pos] = max(left, current)

        # aguarda que as outras threads se sincronizem
        cuda.syncthreads()

        # também utilizado para evitar buffer overflow
        if pos < array.size:
            current = array[pos]
            right = array[pos + 1]

            array[pos] = min(current, right)
            array[pos + 1] = max(current, right)

        # aguarda que as outras threads se sincronizem
        cuda.syncthreads()


def sort(array):
    """
    wrapper para chamar a função de gpu
    """
    stream = cuda.stream() # streams operam assincronamente
    cuda_array = cuda.to_device(array, stream=stream) # leva o array para o dispositivo

    # para cada duas posições disparamos uma thread de execução
    threadsperblock = 512
    blockspergrid = (an_array.size + (threadsperblock - 1)) // threadsperblock // 2

    sort_kernel[blockspergrid, threadsperblock](cuda_array)

    # copia o array novamente para a memoria
    cuda_array.copy_to_host(array, stream=stream)

    # sincroniza tudo =)
    stream.synchronize()

    return array


vec_size = 100000
an_array = np.array(np.random.sample(vec_size), dtype=np.float64)
result = sort(an_array)
print(result)
