import argparse

# 실행하고 싶으면, .py로 만들고 shell에서 실행해야 argparse가 사용되는 모습을 볼 수 이씀
# 현재 파일명은 prog.py이고 shell에서 python prog.py를 실행해야한다

parser = argparse.ArgumentParser(description ='Process some integers.')

# python prog.py -h 이러한 명령어를 실행하게 되면, description 처럼 str 형태로 저장해놓은 놈들이 출력된다.
# 각각의 argument마다 설명을 추가할 수 있으며, -h 명령을 실행하면 나타나게 됨.
# 내부에 들어가는 파라미터에 대한 것은 일단 생략

# 정리하면, argparse는 shell에서 model에 어떤 인자들을 넘겨주고 싶을 때 사용하는 함수이다.

parser.add_argument("integers", metavar='N', type=int,nargs='+',
                    help='an integer for the accumulator')

parser.add_argument('--sum', dest='accumulate', action='store_const',
                    const=sum, default=max,
                    help='sum the integers (default: find the max)')

                    # help는 각 명령어에 대한 설명을 넣어주는 것.
                    # add_argument를 하는 것은 어떻게 하는 거징? 안에 있는 파라미터들이 어떻게 구성되는건지 잘 모르게따.


args = parser.parse_args()
print(args.accumulate(args.integers))

#