

import numpy as np
import pyglet
import time

pyglet.clock.set_fps_limit(10000)

class car_env():
    viewer=None
    action_dim=1
    state_dim=1
    dt=0.1


    def __init__(self):
        self.actions=[0,1,2]
        self.velocity={0:3.,1:5.,2:7.}
        self.terminal=None
        self.breakdown=False
        # node1 (x, y, r, w, l)
        self.car_info=np.array([10,250,0,20,40],dtype=np.float32)
        self.feed_rate=40


    def render(self):
        if self.viewer is None:
            self.viewer=Viewer(self.car_info)
        self.viewer.render()

    def reset(self):
        self.terminal=False
        self.breakdown=False
        self.car_info[:3]=np.array([10,250,0])
        return self._get_observation()

    def step(self,action):
        action=self.actions[action]
        self.car_info[:2] = self.car_info[:2]+\
                            self.velocity[action]*self.dt * np.array([np.cos(self.car_info[2]), np.sin(self.car_info[2])])
        s=self._get_observation()
        self._get_feed_rate(velocity=self.velocity[action])
        r=0
        # print(self.feed_rate)
        if self.feed_rate < 20:
            r=-5
        elif self.feed_rate>50:
            self.breakdown=True
            r=-100
        if self.car_info[0]+self.car_info[4]/2>500:
            self.terminal=True
            r=100
        return s,r,self.terminal,self.breakdown

    def sample_action(self):
        a=np.random.choice(list(range(3)))
        return a

    def set_fps(self, fps=30):
        pyglet.clock.set_fps_limit(fps)

    def _get_observation(self): # 获得5米远的作物密度
        if self.car_info[0]+self.car_info[4]/2+5>500:
            return 6
        else:
            return self.get_crop_density(self.car_info[0]+self.car_info[4]/2+5)

    def get_crop_density(self,x):
        return 2*np.random.rand()*np.sin(x*np.pi/500)+6

    def _get_feed_rate(self,velocity):
        crop_density=self.get_crop_density(self.car_info[0]+self.car_info[4]/2)
        self.feed_rate=crop_density*velocity

    def get_positon(self):
        return self.car_info[0]+self.car_info[4]/2

class Viewer(pyglet.window.Window):
    bar_thc = 5
    fps_display=pyglet.clock.ClockDisplay()

    def __init__(self,car_info):
        super(Viewer, self).__init__(width=500,height=500,resizable=False,caption='car',vsync=True)
        pyglet.gl.glClearColor(1,1,1,1)
        self.car_info=car_info
        self.batch=pyglet.graphics.Batch()
        self.glass=self.batch.add(
            4,pyglet.gl.GL_QUADS,None,
            ('v2f',[0,100,0,400,500,400,500,100]),
            ('c3B',(0,139,0)*4,)
        )
        self.car=self.batch.add(
            4,pyglet.gl.GL_QUADS,None,
            ('v2f',[car_info[0]-car_info[4]/2,car_info[1]-car_info[3]/2,
                    car_info[0]-car_info[4]/2,car_info[1]+car_info[3]/2,
                    car_info[0]+car_info[4]/2,car_info[1]+car_info[3]/2,
                    car_info[0]+car_info[4]/2,car_info[1]-car_info[3]/2]),
            ('c3B',(249,86,86)*4,)
        )# car


    def render(self):
        self._update_position()
        self.switch_to()
        self.dispatch_events()
        self.dispatch_event('on_draw')
        self.flip()

    def on_draw(self):
        self.clear()
        self.batch.draw()

    def _update_position(self):
        (cx,cy,r,w,l)=self.car_info
        xys=[
            [cx+l/2,cy+w/2],
            [cx-l/2,cy+w/2],
            [cx-l/2,cy-w/2],
            [cx+l/2,cy-w/2],
        ]
        r_xys=[]
        for x,y in xys:
            tempX=x-cx
            tempY=y-cy
            rotatedX=tempX*np.cos(r)-tempY*np.sin(r)
            rotatedY=tempX*np.sin(r)+tempY*np.cos(r)

            x=rotatedX+cx
            y=rotatedY+cy
            r_xys += [x,y]
        self.car.vertices=r_xys

if __name__=='__main__':
    env=car_env()
    env.set_fps(5)
    count=0
    while True:
        s=env.reset()
        count += 1
        while True:
            env.render()
            s,r,done,breakdown=env.step(env.sample_action())
            time.sleep(0.005)
            print(count)
            if done or breakdown:
                break
