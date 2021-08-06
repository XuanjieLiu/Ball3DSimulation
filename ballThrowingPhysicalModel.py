import math

GRAVITY = 10.0
BOUNCE_ENERGY_LOSS = 0.19

ACCURACY_HEIGHT = 3
ACCURACY_TIME = 4
ACCURACY_VELOCITY = 3

def uniformLinerCalc(s, v, dt):
    return s + v * dt, v


def highestPosition(h, v):
    time = v / GRAVITY
    height = h + 0.5 * math.pow(v, 2.0) / GRAVITY
    return round(height, ACCURACY_HEIGHT), 0, round(time, ACCURACY_TIME)


def fastestVelocity(h, v):
    velocity = math.sqrt(2 * GRAVITY * h + math.pow(v, 2.0))
    time = (velocity + v) / GRAVITY
    return 0, round(velocity, ACCURACY_VELOCITY), round(time, ACCURACY_TIME)

def velocityAfterEnergyLoss(v):
    return round(v * math.sqrt(1-BOUNCE_ENERGY_LOSS), ACCURACY_VELOCITY)

def distanceCalc(v, dt):
    return (v + v - dt * GRAVITY) * dt / 2

def verticalCalc(h, v, dt):
    if (h == 0 and v == 0): return 0.0, 0.0
    timeRemain = dt
    height = h
    velocity = v
    isUp = velocity > 0
    iter = 0
    while True:
        if isUp:
            nextHeight, nextVelocity, timeCost = highestPosition(height,velocity)
            isUp = False
        else:
            nextHeight, nextVelocity, timeCost = fastestVelocity(height, velocity)
            nextVelocity = velocityAfterEnergyLoss(nextVelocity)
            isUp = True
        iter += 1
        # print(iter, nextHeight, nextVelocity, timeCost, isUp)
        if timeRemain - timeCost <= 0:
            dh = distanceCalc(velocity, timeRemain)
            return height + dh, velocity - timeRemain * GRAVITY
        else:
            timeRemain -= timeCost
            height = nextHeight
            velocity = nextVelocity

def ballNextState(sXYZ, vXYZ, dt):
    nsX, nvX = uniformLinerCalc(sXYZ[0], vXYZ[0], dt)
    nsY, nvY = verticalCalc(sXYZ[1], vXYZ[1], dt)
    nsZ, nvZ = uniformLinerCalc(sXYZ[2], vXYZ[2], dt)
    return [nsX, nsY, nsZ], [nvX, nvY, nvZ]

# sXYZ = [0,10,0]
# vXYZ = [0,0,0]
# dt = 100
#
# [a, b, c], [d, e, f] = ballNextState(sXYZ, vXYZ, dt)
# print(a, b, c, d, e, f)