from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import numpy as np
import time
import random
from ballThrowingPhysicalModel import ballNextState
import cv2


WIN_W = 120
WIN_H = 120
IMG_ROOT_PATH = 'Ball3DImg'
IMG_SAVE_PATH = f'{IMG_ROOT_PATH}/{WIN_W}_{WIN_H}'
MAKE_IMG = True
DRAW_GIRD = False

class BallViewer:
    def __init__(self):
        # 小球位置信息
        self.sX = 0
        self.sY = 1
        self.sZ = 0

        # 小球速度信息
        self.vX, self.vY, self.vZ = self.randomBallInitVelocity()

        # 小球半径
        self.ballRadius = 0.5

        # opengl视角信息
        self.IS_PERSPECTIVE = True  # 透视投影
        self.VIEW = np.array([-0.8, 0.8, -0.8, 0.8, 1.0, 10.0])  # 视景体的left/right/bottom/top/near/far六个面
        self.SCALE_K = np.array([1.0, 1.0, 1.0])  # 模型缩放比例
        self.EYE = np.array([0.0, 4.0, 2.0])  # 眼睛的位置（默认z轴的正方向）
        self.LOOK_AT = np.array([0.0, 0.0, -5.0])  # 瞄准方向的参考点（默认在坐标原点）
        self.EYE_UP = np.array([0.0, 1.0, 0.0])  # 定义对观察者而言的上方（默认y轴的正方向）
        self.DIST, self.PHI, self.THETA = self.getposture()  # 眼睛与观察目标之间的距离、仰角、方位角
        self.WIN_W, self.WIN_H = WIN_W, WIN_H  # 保存窗口宽度和高度的变量

        # 鼠标操作信息
        self.LEFT_IS_DOWNED = False  # 鼠标左键被按下
        self.MOUSE_X, self.MOUSE_Y = 0, 0  # 考察鼠标位移量时保存的起始位置

        self.openGLInit()
        self.currTime = time.time()
        self.lastScreenShot = time.time() # 上次截图时间d
        self.timeInOneTest = 0


    def getposture(self):
        dist = np.sqrt(np.power((self.EYE - self.LOOK_AT), 2).sum())
        if dist > 0:
            phi = np.arcsin((self.EYE[1] - self.LOOK_AT[1]) / dist)
            theta = np.arcsin((self.EYE[0] - self.LOOK_AT[0]) / (dist * np.cos(phi)))
        else:
            phi = 0.0
            theta = 0.0
        return dist, phi, theta


    def init(self):
        glClearColor(0.4, 0.4, 0.4, 1.0)  # 设置画布背景色。注意：这里必须是4个参数
        glEnable(GL_DEPTH_TEST)  # 开启深度测试，实现遮挡关系
        glDepthFunc(GL_LEQUAL)  # 设置深度测试函数（GL_LEQUAL只是选项之一）-
        glEnable(GL_LIGHT0) # 启用0号光源
        glLightfv(GL_LIGHT0, GL_POSITION, GLfloat_4(0, 1, 4, 0)) # 设置光源的位置
        glLightfv(GL_LIGHT0, GL_SPOT_DIRECTION, GLfloat_3(0, 0, -1)) # 设置光源的照射方向
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE) # 设置材质颜色
        glEnable(GL_COLOR_MATERIAL)

    def initRender(self):
        # 清除屏幕及深度缓存
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # 设置投影（透视投影）
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()

        if self.WIN_W > self.WIN_H:
            if self.IS_PERSPECTIVE:
                glFrustum(self.VIEW[0] * self.WIN_W / self.WIN_H, self.VIEW[1] * self.WIN_W / self.WIN_H, self.VIEW[2], self.VIEW[3], self.VIEW[4], self.VIEW[5])
            else:
                glOrtho(self.VIEW[0] * self.WIN_W / self.WIN_H, self.VIEW[1] * self.WIN_W / self.WIN_H, self.VIEW[2], self.VIEW[3], self.VIEW[4], self.VIEW[5])
        else:
            if self.IS_PERSPECTIVE:
                glFrustum(self.VIEW[0], self.VIEW[1], self.VIEW[2] * self.WIN_H / self.WIN_W, self.VIEW[3] * self.WIN_H / self.WIN_W, self.VIEW[4], self.VIEW[5])
            else:
                glOrtho(self.VIEW[0], self.VIEW[1], self.VIEW[2] * self.WIN_H / self.WIN_W, self.VIEW[3] * self.WIN_H / self.WIN_W, self.VIEW[4], self.VIEW[5])

        # 设置模型视图
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        # 几何变换
        glScale(self.SCALE_K[0], self.SCALE_K[1], self.SCALE_K[2])

        # 设置视点
        gluLookAt(
            self.EYE[0], self.EYE[1], self.EYE[2],
            self.LOOK_AT[0], self.LOOK_AT[1], self.LOOK_AT[2],
            self.EYE_UP[0], self.EYE_UP[1], self.EYE_UP[2]
        )

        # 设置视口
        glViewport(0, 0, self.WIN_W, self.WIN_H)


    def drawGird(self):
        glBegin(GL_LINES)
        glColor4f(0.0, 0.0, 0.0, 1)  # 设置当前颜色为黑色不透明
        for i in range(401):
            glVertex3f(-100.0 + 0.5 * i, -self.ballRadius, -100)
            glVertex3f(-100.0 + 0.5 * i, -self.ballRadius, 100)
            glVertex3f(-100.0, -self.ballRadius, -100 + 0.5 * i)
            glVertex3f(100.0, -self.ballRadius, -100 + 0.5 * i)
        glEnd()
        glLineWidth(1)

    def drawBall(self):
        dt = time.time() - self.currTime
        print(dt)
        self.currTime += dt
        self.timeInOneTest += dt
        [self.sX, self.sY, self.sZ], [self.vX, self.vY, self.vZ] = ballNextState([self.sX, self.sY, self.sZ], [self.vX, self.vY, self.vZ], dt)
        if self.timeInOneTest > 4:
            self.timeInOneTest = 0
            self.resetBallPosition()
            self.vX, self.vY, self.vZ = self.randomBallInitVelocity()

        glPushMatrix()
        glColor3f(0.05, 0.9, 0.05)
        glTranslatef(self.sX, self.sY, -self.sZ)  # Move to the place
        quad = gluNewQuadric()
        gluSphere(quad, self.ballRadius, 90, 90)
        gluDeleteQuadric(quad)
        glPopMatrix()

        if MAKE_IMG and (self.currTime - self.lastScreenShot > 0.3):
            self.screenShot(self.WIN_W, self.WIN_H, str(round(self.currTime, 1)))
            self.lastScreenShot = self.currTime



    def randomBallInitVelocity(self):
        return random.random() * 4 - 2, (random.random() + 0.2) * 20 - 10, random.random() * 2

    def resetBallPosition(self):
        self.sX = 0
        self.sY = 1
        self.sZ = 0

    def draw(self):
        self.initRender()
        glEnable(GL_LIGHTING)  # 启动光照
        if DRAW_GIRD:
            self.drawGird()
        self.drawBall()
        glDisable(GL_LIGHTING)  # 每次渲染后复位光照状态

        # 把数据刷新到显存上
        glFlush()
        glutSwapBuffers()  # 切换缓冲区，以显示绘制内容


    def reshape(self, width, height):
        self.WIN_W, self.WIN_H = width, height
        glutPostRedisplay()

    def mouseclick(self, button, state, x, y):
        self.MOUSE_X, self.MOUSE_Y = x, y
        if button == GLUT_LEFT_BUTTON:
            self.LEFT_IS_DOWNED = state == GLUT_DOWN
        elif button == 3:
            self.SCALE_K *= 1.05
            glutPostRedisplay()
        elif button == 4:
            self.SCALE_K *= 0.95
            glutPostRedisplay()

    def mousemotion(self, x, y):
        if self.LEFT_IS_DOWNED:
            dx = self.MOUSE_X - x
            dy = y - self.MOUSE_Y
            self.MOUSE_X, self.MOUSE_Y = x, y

            self.PHI += 2 * np.pi * dy / self.WIN_H
            self.PHI %= 2 * np.pi
            self.THETA += 2 * np.pi * dx / self.WIN_W
            self.THETA %= 2 * np.pi
            r = self.DIST * np.cos(self.PHI)

            self.EYE[1] = self.DIST * np.sin(self.PHI)
            self.EYE[0] = r * np.sin(self.THETA)
            self.EYE[2] = r * np.cos(self.THETA)

            if 0.5 * np.pi < self.PHI < 1.5 * np.pi:
                self.EYE_UP[1] = -1.0
            else:
                self.EYE_UP[1] = 1.0

            glutPostRedisplay()

    def keydown(self, key, x, y):
        if key in [b'x', b'X', b'y', b'Y', b'z', b'Z']:
            if key == b'x':  # 瞄准参考点 x 减小
                self.LOOK_AT[0] -= 0.01
            elif key == b'X':  # 瞄准参考 x 增大
                self.LOOK_AT[0] += 0.01
            elif key == b'y':  # 瞄准参考点 y 减小
                self.LOOK_AT[1] -= 0.01
            elif key == b'Y':  # 瞄准参考点 y 增大
                self.LOOK_AT[1] += 0.01
            elif key == b'z':  # 瞄准参考点 z 减小
                self.LOOK_AT[2] -= 0.01
            elif key == b'Z':  # 瞄准参考点 z 增大
                self.LOOK_AT[2] += 0.01

            self.DIST, self.PHI, self.THETA = self.getposture()
            glutPostRedisplay()
        elif key == b'\r':  # 回车键，视点前进
            self.EYE = self.LOOK_AT + (self.EYE - self.LOOK_AT) * 0.9
            self.DIST, self.PHI, self.THETA = self.getposture()
            glutPostRedisplay()
        elif key == b'\x08':  # 退格键，视点后退
            self.EYE = self.LOOK_AT + (self.EYE - self.LOOK_AT) * 1.1
            self.DIST, self.PHI, self.THETA = self.getposture()
            glutPostRedisplay()
        elif key == b' ':  # 空格键，切换投影模式
            self.IS_PERSPECTIVE = not self.IS_PERSPECTIVE
            glutPostRedisplay()

    def openGLInit(self):
        glutInit()
        displayMode = GLUT_DOUBLE | GLUT_ALPHA | GLUT_DEPTH
        glutInitDisplayMode(displayMode)

        glutInitWindowSize(self.WIN_W, self.WIN_H)
        glutInitWindowPosition(300, 50)
        glutCreateWindow('Ball Throwing Simulation')

        self.init()  # 初始化画布
        glutDisplayFunc(self.draw)  # 注册回调函数draw()
        glutIdleFunc(self.draw)
        glutReshapeFunc(self.reshape)  # 注册响应窗口改变的函数reshape()
        glutMouseFunc(self.mouseclick)  # 注册响应鼠标点击的函数mouseclick()
        glutMotionFunc(self.mousemotion)  # 注册响应鼠标拖拽的函数mousemotion()
        glutKeyboardFunc(self.keydown)  # 注册键盘输入的函数keydown()

    def mainLoop(self):
        glutMainLoop()  # 进入glut主循环

    def screenShot(self, w, h, imgName):
        glReadBuffer(GL_FRONT)
        # 从缓冲区中的读出的数据是字节数组
        data = glReadPixels(0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE)
        arr = np.zeros((h * w * 3), dtype=np.uint8)
        for i in range(0, len(data), 3):
            # 由于opencv中使用的是BGR而opengl使用的是RGB所以arr[i] = data[i+2]，而不是arr[i] = data[i]
            arr[i] = data[i + 2]
            arr[i + 1] = data[i + 1]
            arr[i + 2] = data[i]
        arr = np.reshape(arr, (h, w, 3))
        # 因为opengl和OpenCV在Y轴上是颠倒的，所以要进行垂直翻转，可以查看cv2.flip函数
        cv2.flip(arr, 0, arr)
        cv2.imshow('scene', arr)
        cv2.imwrite(f'{IMG_SAVE_PATH}/{imgName}.png', arr)  # 写入图片
        cv2.waitKey(1)


if __name__ == "__main__":
    if MAKE_IMG:
        if not os.path.isdir(IMG_ROOT_PATH):
            os.mkdir(IMG_ROOT_PATH)
        if not os.path.isdir(IMG_SAVE_PATH):
            os.mkdir(IMG_SAVE_PATH)
    ballView = BallViewer()
    ballView.mainLoop()

