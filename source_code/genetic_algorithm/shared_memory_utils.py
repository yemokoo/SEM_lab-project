import pandas as pd
import numpy as np
from multiprocessing import shared_memory
import pickle # 문자열 컬럼 직렬화를 위해
import os
import time

# 생성된 공유 메모리 객체들을 추적 (메인 프로세스에서 정리하기 위함)
# 이 리스트는 fitness_func 내에서 관리되도록 수정하는 것이 더 안전합니다.
# 여기서는 설명을 위해 전역으로 두지만, 실제로는 함수의 반환값으로 관리하세요.
# _active_shm_objects = [] 

def put_df_to_shared_memory(df, unique_prefix=""):
    shm_infos = []
    shm_objects = []

    for col_name in df.columns:
        column = df[col_name]
        shm = None # try-except-finally 용
        try:
            if pd.api.types.is_numeric_dtype(column) or pd.api.types.is_bool_dtype(column):
                np_array = column.to_numpy()
                col_nbytes = np_array.nbytes
                original_dtype = np_array.dtype
                original_shape = np_array.shape
                is_pickled = False
                
                shm_name = f"shm_{unique_prefix}_{os.getpid()}_{col_name.replace(' ', '_')}_{time.time_ns()}"
                shm = shared_memory.SharedMemory(create=True, size=col_nbytes, name=shm_name)
                shared_np_array = np.ndarray(original_shape, dtype=original_dtype, buffer=shm.buf)
                shared_np_array[:] = np_array[:]
                
            elif column.dtype == 'object' or pd.api.types.is_string_dtype(column) or pd.api.types.is_categorical_dtype(column):
                # 문자열/Object/Categorical 타입: pickle로 직렬화
                pickled_series_bytes = pickle.dumps(column.copy()) # 직렬화된 바이트 (copy()로 안전하게)
                col_nbytes = len(pickled_series_bytes) # 실제 바이트 수
                original_dtype = 'object_pickled' 
                original_shape = (len(column),) # 원래 Series의 요소 수
                is_pickled = True

                shm_name = f"shm_{unique_prefix}_{os.getpid()}_{col_name.replace(' ', '_')}_{time.time_ns()}"
                shm = shared_memory.SharedMemory(create=True, size=col_nbytes, name=shm_name)
                shm.buf[:col_nbytes] = pickled_series_bytes # 정확한 바이트만큼 쓰기
            else:
                print(f"Warning: Column '{col_name}' with dtype {column.dtype} not supported for shared memory. Skipping.")
                continue
            
            shm_objects.append(shm) # 정리 대상 추가
            shm_infos.append({
                'name': shm.name,
                'dtype': original_dtype,
                'shape': original_shape,
                'col_name': col_name,
                'is_pickled': is_pickled,
                'nbytes': col_nbytes # 직렬화된 실제 바이트 수 또는 배열 바이트 수
            })
        except Exception as e:
            print(f"Error creating shared memory for column {col_name}: {e}")
            if shm: # 만약 shm 객체가 생성되었다면 정리 시도
                shm.close()
                try:
                    shm.unlink()
                except FileNotFoundError:
                    pass
            # 이미 shm_objects에 추가된 것들도 정리
            for s_obj in shm_objects:
                if s_obj is not shm: # 현재 에러난 shm 제외
                    s_obj.close()
                    try:
                        s_obj.unlink()
                    except FileNotFoundError:
                        pass
            raise

    return shm_infos, shm_objects


def reconstruct_df_from_shared_memory(shm_infos):
    data_for_df = {}
    opened_shms = [] 
    try:
        for info in shm_infos:
            shm = shared_memory.SharedMemory(name=info['name'])
            opened_shms.append(shm)

            if info['is_pickled']:
                pickled_bytes_from_shm = shm.buf[:info['nbytes']].tobytes() 
                data_for_df[info['col_name']] = pickle.loads(pickled_bytes_from_shm) # .copy()는 Series/DF에 내장
            else:
                reconstructed_array = np.ndarray(info['shape'], dtype=info['dtype'], buffer=shm.buf)
                data_for_df[info['col_name']] = reconstructed_array.copy() 

        ordered_cols = [info['col_name'] for info in shm_infos]
        return pd.DataFrame(data_for_df)[ordered_cols]
    finally:
        for shm_obj in opened_shms:
            shm_obj.close()


def cleanup_shms_by_info(shm_infos):
    """메인 프로세스에서 정보 리스트를 기반으로 공유 메모리 정리"""
    print(f"Cleaning up {len(shm_infos)} shared memory blocks by info...")
    for info in shm_infos:
        try:
            # unlink는 이미 닫힌 shm 객체에도 이름으로 가능
            shm_temp = shared_memory.SharedMemory(name=info['name'])
            shm_temp.close() # 혹시 열려있을 수 있으니 close
            shm_temp.unlink()
        except FileNotFoundError:
            print(f"SHM {info['name']} already unlinked or never created properly.")
            pass
        except Exception as e:
            print(f"Error unlinking SHM {info['name']}: {e}")