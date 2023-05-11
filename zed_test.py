import time
import threading
import pyzed.sl as sl
from jetbot import Robot


class Tracker(threading.Thread):
    def __init__(self, ):
        threading.Thread.__init__(self)
        self.cam = sl.Camera()
        init_params = sl.InitParameters()
        init_params.camera_fps = 15
        init_params.camera_resolution = sl.RESOLUTION.HD720
        init_params.coordinate_units = sl.UNIT.METER
        err = self.cam.open(init_params)
        tracking_params = sl.PositionalTrackingParameters()
        self.cam.enable_positional_tracking(tracking_params)
        self.runtime = sl.RuntimeParameters()
        self.pose = sl.Pose()
        self.rot = self.pose.get_rotation_matrix()
        self.tns = self.pose.get_translation()


    def run(self):
        if self.cam.grab(self.runtime) == sl.ERROR_CODE.SUCCESS:
            if self.cam.get_position(self.pose) == sl.POSITIONAL_TRACKING_STATE.OK:
                self.rot = self.pose.get_rotation_matrix()
                self.tns = self.pose.get_translation()
        else:
            print('Unable to get runtime')

    def close(self):
        self.cam.close()


class controller():
    def __init__(self, robot, tracker, speed=0.12, lbias=0.00, rbias=0.00):
        self.jb = robot
        self.tracker = tracker
        self.tracker.run()
        self.lspeed = speed + lbias
        self.rspeed = speed + rbias
        self.pi = 3.14159265359
        self.pi2 = self.pi / 2.0
        self.dist = 0.025*12
        self.actions = [0,1,2,3] #Up, Down, Left, Right
        self.thresh = 0.1
        self.thresh_rot = 0.1
        self.north = self.tracker.rot.get_rotation_vector()[1]
        self.south = self.north - self.pi
        self.east = self.north + self.pi2
        self.west = self.north - self.pi2
        self.rot = self.north


    def left(self, desired=None):
        done = False
        self.tracker.run()
        rot1 = self.tracker.rot.get_rotation_vector()[1]
        self.jb.set_motors(-self.lspeed,self.rspeed)
        while not done:
            self.tracker.run()
            self.rot = self.tracker.rot.get_rotation_vector()[1]
            done = True if abs(rot1 - self.rot) > self.pi2  else False
        self.jb.stop()


    def right(self, desired=None):
        done = False
        self.tracker.run()
        rot1 = self.tracker.rot.get_rotation_vector()[1]
        self.jb.set_motors(self.lspeed,-self.rspeed)
        while not done:
            self.tracker.run()
            self.rot = self.tracker.rot.get_rotation_vector()[1]
            done = True if abs(rot1 - self.rot) > self.pi2 else False
        self.jb.stop()


    def forward(self,):
        done = False
        self.tracker.run()
        tns1 = self.tracker.tns.get()
        self.jb.set_motors(self.lspeed,self.rspeed)
        while not done:
            self.tracker.run()
            tns2 = self.tracker.tns.get()
            done = True if abs(tns1[2] - tns2[2]) > self.dist or abs(tns1[0] - tns2[0]) > self.dist else False
        self.jb.stop()
  

    def rotate(self, desired, left):
        print(f'DESIRED : {desired}')
        done = False
        if left:
            self.jb.set_motors(-self.lspeed,self.rspeed)
        else:
            self.jb.set_motors(self.lspeed,-self.rspeed)
        while not done:
            self.tracker.run()
            rot = self.tracker.rot.get_rotation_vector()[1]
            done = True if abs(desired - rot) < self.thresh_rot else False
            print(f'{rot}')
        
        self.jb.stop()

    def desired_rotation(self, action):
        if action == 0:
            return self.north
        elif action == 1:
            return self.south
        elif action == 2:
            return self.west
        else:
            return self.east


    def move(self, action):
        if action in self.actions:
            self.tracker.run()
            desired_rot = self.desired_rotation(action)
            self.rot = self.tracker.rot.get_rotation_vector()[1]
            if action == 1:
                if self.rot < 0.0:
                    self.rotate(desired_rot, left=True)
                else:
                    self.rotate(desired_rot, left=False)
            elif self.rot > self.east and desired_rot == self.west:
                self.rotate(desired_rot, left=False)
            elif self.rot < self.west and desired_rot == self.east:
                self.rotate(desired_rot, left=True)
            elif self.rot < desired_rot:
                self.rotate(desired_rot, left=False)
            elif self.rot > desired_rot:
                self.rotate(desired_rot, left=True)
            self.forward()
        else:
            print('[WARNING] Invalid action recieved.')


def main():
    path = [2,2,2,0,0,0,3,0,3,3,0,0,2,2,2,1,2,1,2,2,0,0]
    tracker = Tracker()
    jb = controller(Robot(), tracker, lbias=0.005)
    for action in path:
        jb.move(action)
    tracker.close()

if __name__ == "__main__":
    main()
