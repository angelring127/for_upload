import os
from PIL import Image

# 처리할 이미지 확장자들
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png')

# 현재 디렉토리의 모든 파일 검사
for filename in os.listdir('.'):
    # 파일이 이미지인 경우에만 처리
    if filename.lower().endswith(IMAGE_EXTENSIONS):
        try:
            # 이미지 열기
            image = Image.open(filename)
            original_width, original_height = image.size

            # 새로운 너비 설정 (377 픽셀)
            new_width = 377

            # 이미지 자르기 (오른쪽에서부터)
            cropped_image = image.crop((original_width - new_width, 0, original_width, original_height))

            # 원본 이미지 닫기
            image.close()

            # 수정된 이미지를 원본 파일로 저장 (덮어쓰기)
            cropped_image.save(filename, quality=95)
            print(f'처리 완료: {filename}')
            
        except Exception as e:
            print(f'에러 발생 ({filename}): {str(e)}')

print('모든 이미지 처리가 완료되었습니다.')
