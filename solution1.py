from gym_duckietown.tasks.task_solution import TaskSolution
import numpy as np
import cv2


class DontCrushDuckieTaskSolution(TaskSolution):
    def __init__(self, generated_task):
        super().__init__(generated_task)

    def solve(self):
        env = self.generated_task['env']
        # getting the initial picture
        obs, _, _, _ = env.step([0, 0])
        # convect in for work with cv
        img = cv2.cvtColor(np.ascontiguousarray(obs), cv2.COLOR_BGR2RGB)

        # add here some image processing

        condition = True
        while condition:
            obs, reward, done, info = env.step([1, 0])
            img = cv2.cvtColor(np.ascontiguousarray(obs), cv2.COLOR_BGR2RGB)

            img2 = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            bnw = cv2.inRange(img2, (0, 150, 170), (0, 220, 255))
            contours, ret = cv2.findContours(bnw, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            if contours:
                contours = sorted(contours, key=cv2.contourArea, reverse=True)
                contours = contours[0]
                (x, y, w, h) = cv2.boundingRect(contours)
                print(w, h)
                if h > 150:
                    condition = False
                    break

            # add here some image processing
            condition = True
            env.render()