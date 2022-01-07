# sin과 cos 함수를 그려주기 위해 math 함수를 선언
import math
# tensorboard에 data를 올릴려면 SummaryWriter를 선언 해줘야 합니다.
from tensorboardX import SummaryWriter

writer = SummaryWriter(logdir='scalar/sin&cos')

for step in range(-360, 360):
    angle_rad = step * math.pi / 180
    writer.add_scalar('sin', math.sin(angle_rad), step)
    writer.add_scalar('cos', math.cos(angle_rad), step)
    
writer.close()