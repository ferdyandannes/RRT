"""

Path planning Sample Code with RRT*

author: Atsushi Sakai(@Atsushi_twi)

"""
import sys
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
import cv2

import math
import os
import sys
import h5py

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../RRT/")

try:
    from rrt import RRT
except ImportError:
    raise

show_animation = True

def check_dir(dir_list):
    for d in dir_list:
        if not os.path.isdir(d):
            print('Create directory :\n' + d)
            os.makedirs(d)


class RRTStar(RRT):
    """
    Class for RRT Star planning
    """

    class Node(RRT.Node):
        def __init__(self, x, y):
            super().__init__(x, y)
            self.cost = 0.0

    def __init__(self, start, goal, obstacle_list, rand_area,
                 expand_dis=3.0,
                 path_resolution=0.1,
                 goal_sample_rate=2,
                 max_iter=200,
                 connect_circle_dist=100.0 # 50
                 ):
        super().__init__(start, goal, obstacle_list,
                         rand_area, expand_dis, path_resolution, goal_sample_rate, max_iter)
        """
        Setting Parameter

        start:Start Position [x,y]
        goal:Goal Position [x,y]
        obstacleList:obstacle Positions [[x,y,size],...]
        randArea:Random Sampling Area [min,max]

        """
        self.connect_circle_dist = connect_circle_dist
        self.goal_node = self.Node(goal[0], goal[1])

    def planning(self, animation=True, search_until_max_iter=True):
        """
        rrt star path planning

        animation: flag for animation on or off
        search_until_max_iter: search until max iteration for path improving or not
        """

        self.node_list = [self.start]
        for i in range(self.max_iter):
            #print("Iter:", i, ", number of nodes:", len(self.node_list))
            rnd = self.get_random_node()
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd)
            new_node = self.steer(self.node_list[nearest_ind], rnd, self.expand_dis)

            if self.check_collision(new_node, self.obstacle_list):
                near_inds = self.find_near_nodes(new_node)
                new_node = self.choose_parent(new_node, near_inds)
                if new_node:
                    self.node_list.append(new_node)
                    self.rewire(new_node, near_inds)

            if animation and i % 5 == 0:
                self.draw_graph(rnd)

            if (not search_until_max_iter) and new_node:  # check reaching the goal
                last_index = self.search_best_goal_node()
                if last_index:
                    return self.generate_final_course(last_index)

        print("reached max iteration")

        last_index = self.search_best_goal_node()
        if last_index:
            return self.generate_final_course(last_index)

        return None

    def choose_parent(self, new_node, near_inds):
        if not near_inds:
            return None

        # search nearest cost in near_inds
        costs = []
        for i in near_inds:
            near_node = self.node_list[i]
            t_node = self.steer(near_node, new_node)
            if t_node and self.check_collision(t_node, self.obstacle_list):
                costs.append(self.calc_new_cost(near_node, new_node))
            else:
                costs.append(float("inf"))  # the cost of collision node
        min_cost = min(costs)

        if min_cost == float("inf"):
            print("There is no good path.(min_cost is inf)")
            return None

        min_ind = near_inds[costs.index(min_cost)]
        new_node = self.steer(self.node_list[min_ind], new_node)
        new_node.parent = self.node_list[min_ind]
        new_node.cost = min_cost

        return new_node

    def search_best_goal_node(self):
        dist_to_goal_list = [self.calc_dist_to_goal(n.x, n.y) for n in self.node_list]
        goal_inds = [dist_to_goal_list.index(i) for i in dist_to_goal_list if i <= self.expand_dis]

        safe_goal_inds = []
        for goal_ind in goal_inds:
            t_node = self.steer(self.node_list[goal_ind], self.goal_node)
            if self.check_collision(t_node, self.obstacle_list):
                safe_goal_inds.append(goal_ind)

        if not safe_goal_inds:
            return None

        min_cost = min([self.node_list[i].cost for i in safe_goal_inds])
        for i in safe_goal_inds:
            if self.node_list[i].cost == min_cost:
                return i

        return None

    def find_near_nodes(self, new_node):
        nnode = len(self.node_list) + 1
        r = self.connect_circle_dist * math.sqrt((math.log(nnode) / nnode))
        dist_list = [(node.x - new_node.x) ** 2 +
                     (node.y - new_node.y) ** 2 for node in self.node_list]
        near_inds = [dist_list.index(i) for i in dist_list if i <= r ** 2]
        return near_inds

    def rewire(self, new_node, near_inds):
        for i in near_inds:
            near_node = self.node_list[i]
            edge_node = self.steer(new_node, near_node)
            if not edge_node:
                continue
            edge_node.cost = self.calc_new_cost(new_node, near_node)

            no_collision = self.check_collision(edge_node, self.obstacle_list)
            improved_cost = near_node.cost > edge_node.cost

            if no_collision and improved_cost:
                near_node = edge_node
                near_node.parent = new_node
                self.propagate_cost_to_leaves(new_node)

    def calc_new_cost(self, from_node, to_node):
        d, _ = self.calc_distance_and_angle(from_node, to_node)
        return from_node.cost + d

    def propagate_cost_to_leaves(self, parent_node):

        for node in self.node_list:
            if node.parent == parent_node:
                node.cost = self.calc_new_cost(parent_node, node)
                self.propagate_cost_to_leaves(node)


def main(data_dir):
    print("Start " + __file__)

    free_area_dir = os.path.join(data_dir, 'Free_Area/')
    free = os.listdir(free_area_dir)
    free.sort()

    path_dir = data_dir+'Path/'
    check_dir([path_dir])

    for frame_id in range(len(free)):        # for frame_id in range(len(free)):
        num = str(frame_id).zfill(4)
        print("frame = ", num)

        # Create path in txt
        save_path_txt = os.path.join(path_dir, num+'.txt')
        position = open(save_path_txt, 'w+')

        with h5py.File(free_area_dir+num+'.h5','r') as fa:
            frame_id = fa['frame_id'].value
            range_const = fa['range_const'].value
            pos_x = fa['pos_x'].value
            pos_y = fa['pos_y'].value
            check_pos = fa['check_pos'].value

        print("check_pos = ", check_pos)

        nrow = len(check_pos)
        ncol = len(check_pos[0])

        print("nrow = ", nrow)
        print("ncol = ", ncol)

        # Scan the start and end
        # Start
        start_pos = []
        for j in range(0, ncol):
            if check_pos[nrow-1, j] == 1:
                start_pos.append([j, 0])

        if len(start_pos) == 0:
            for j in range(0, ncol):
                if check_pos[nrow-2, j] == 1:
                    start_pos.append([j, 1])

        if len(start_pos) == 0:
            for j in range(0, ncol):
                if check_pos[nrow-3, j] == 1:
                    start_pos.append([j, 2])

        # End
        end_pos = []
        for j in range(0, ncol):
            if check_pos[0, j] == 1:
                end_pos.append([j, nrow-1])

        for j in range(0, ncol):
            if check_pos[1, j] == 1:
                end_pos.append([j, nrow-2])

        # for j in range(0, ncol):
        #     if check_pos[2, j] == 1:
        #         end_pos.append([j, nrow-3])

        # if len(end_pos) == 0:
        #     for j in range(0, ncol):
        #         if check_pos[1, j] == 1:
        #             end_pos.append([j, nrow-2])

        if len(end_pos) == 0:
            for j in range(0, ncol):
                if check_pos[2, j] == 1:
                    end_pos.append([j, nrow-3])

        print("start_pos = ", start_pos)
        print("end_pos = ", end_pos)

        # Obstacle Plan --> Kebalik
        halangan = 0.7
        obstacle_list = []
        for i in range(0, nrow):
                for j in range(0, ncol):
                    if check_pos[i, j] == 0:
                        i_mod = i - (nrow-1)
                        i_mod = abs(i_mod)
                        obstacle_list.append([j,i_mod,halangan])

        # Loop over the starting point
        save_path = []
        for i in range(len(start_pos)):
            for j in range(len(end_pos)):
                titik_awal = start_pos[i]
                titik_akhir = end_pos[j]
                rrt_star = RRTStar(start=titik_awal,
                                   goal=titik_akhir,
                                   rand_area=[0, ncol-0.5],
                                   obstacle_list=obstacle_list)

                path = rrt_star.planning(animation=show_animation)

                print("jjjjjjjjjjj = ", j)

                if path is None:
                    print("Cannot find path")
                else:
                    print("found path!!")
                    print("path = ", path)

                    save_path.append(path)

                    # Draw final path
                    if show_animation:
                        rrt_star.draw_graph()
                        plt.plot([x for (x, y) in path], [y for (x, y) in path], '-r')
                        plt.grid(True)
                        plt.xlim(0,ncol-0.5)
                        plt.ylim(0,nrow)
                        
                        #plt.pause(0.01)  # Need for Mac
                        #plt.show()

                        plt.show(block=False)
                        plt.pause(0.01)
                        plt.close()

                print("")

        print("save_path = ", save_path)

        # save_path = np.array(save_path)
        # panjang = len(save_path)
        # dt = h5py.special_dtype(vlen=np.dtype('float64'))
        # path_file.create_dataset('save_path', (panjang,), dtype=dt)
        # path_file['save_path'][...] = save_path

        #save_pathh = np.array(save_path)
        #path_file.create_dataset('save_path', data = save_pathh)
        print("")

        for x in range(len(save_path)):
            path = save_path[x]
            print("path = ", path)
            for y in range(len(path)):
                path_child = path[y]
                path_tulis = str(path_child[0]) + " " + str(path_child[1]) + " "
                position.write(path_tulis)
            position.write("\n")

        position.close()
        plt.close()

def pilih(data_dir):
    path_dir = os.path.join(data_dir, 'Path/')
    path = os.listdir(path_dir)
    path.sort()

    coba_dir = os.path.join(data_dir, 'Coba/')
    coba = os.listdir(coba_dir)
    coba.sort()

    final_path_dir = data_dir+'Final_Path/'
    check_dir([final_path_dir])

    for frame_id in range(len(path)):       # for frame_id in range(len(path)):
        nama_file = str(frame_id).zfill(4)

        print("file = ", nama_file)

        fp_file = h5py.File(os.path.splitext(final_path_dir+coba[frame_id])[0]+'.h5','w')

        # Read Path dari RRT (.txt file)
        baca_path = os.path.join(path_dir,nama_file+".txt")
        with open(baca_path) as baca:
            baca_info = baca.readlines()

        print("baca_info = ", baca_info)

        save_path = []
        for q in range(len(baca_info)):
            isi_path = []
            info = baca_info[q].strip().split()
            print("info = ", info)
            path_x = info[::2]
            path_y = info[1::2]

            for w in range(len(path_x)):
                ind_path_x = path_x[w]
                ind_path_y = path_y[w]

                isi_path.append([float(ind_path_x), float(ind_path_y)])

            save_path.append(isi_path)

            print("")

        print("save_path = ", save_path)

        # Untuk baca path
        tot_x = []
        tot_y = []
        for x in range(len(save_path)):
            path = save_path[x]
            pos_x = []
            pos_y = []
            for y in range(len(path)):
                path_child = path[y]
                x = path_child[0]
                y = path_child[1]
                print("x = ", x)
                print("y = ", y)
                pos_x.append(x)
                pos_y.append(y)

            tot_x.append(pos_x)
            tot_y.append(pos_y)

        #print("")
        #print("tot_x = ", tot_x)
        #print("tot_y = ", tot_y)
        #print("")

        # Baca Info untuk visualisasi
        with h5py.File(coba_dir+nama_file+'.h5','r') as ra:
            inv_save_x = ra['pos_x_rev'].value
            inv_save_y = ra['pos_y_rev'].value
            check_pos = ra['check_pos'].value
            range_const = ra['range_const'].value
            list_save = ra['list_save'].value
            tengah = ra['tengah'].value
        
        final_path = np.zeros((6,list_save))

        # Untuk mapping dari path ke posisi jalan
        for i in range(len(tot_x)):
            baca_x = tot_x[i]
            baca_y = tot_y[i]

            maksimal = len(baca_x)
            #print("maksimal = ", maksimal)

            for j in range(len(baca_x)):
                if j == 0:
                    old_x = baca_x[j]
                    old_y = baca_y[j]
                elif j > 0:
                    cur_x = baca_x[j]
                    cur_y = baca_y[j]
                    x1 = old_x
                    y1 = old_y
                    x2 = cur_x
                    y2 = cur_y

                    beda_x = cur_x - old_x
                    beda_y = cur_y - old_y
                    
                    print("beda_x = ", beda_x)
                    print("beda_y = ", beda_y)

                    # Berdasarkan X
                    if beda_x > 0:
                        for nilai_x in np.arange(old_x, cur_x,0.1):
                            # Persamaan garis lurus
                            nilai_y = (nilai_x*(y2 - y1) - (x1*y2) + (x2*y1))/(x2 - x1)
                            #print("nilai_x = ", nilai_x)
                            #print("nilai_y = ", nilai_y)

                            # Mapping X
                            if -0.5 <= nilai_x <= 0.5:
                                pos_x = 0
                            elif 0.5 < nilai_x <= 1.5:
                                pos_x = 1
                            elif 1.5 < nilai_x <= 2.5:
                                pos_x = 2
                            elif 2.5 < nilai_x <= 3.5:
                                pos_x = 3
                            elif 3.5 < nilai_x <= 4.5:
                                pos_x = 4
                            elif 4.5 < nilai_x <= 5.5:
                                pos_x = 5
                            elif 5.5 < nilai_x <= 6.5:
                                pos_x = 6

                            # Mapping Y
                            if 0 <= nilai_y <= 0.5:
                                pos_y = 5
                            elif 0.5 < nilai_y <= 1.5:
                                pos_y = 4
                            elif 1.5 < nilai_y <= 2.5:
                                pos_y = 3
                            elif 2.5 < nilai_y <= 3.5:
                                pos_y = 2
                            elif 3.5 < nilai_y <= 4.5:
                                pos_y = 1
                            elif 4.5 < nilai_y <= 5.5:
                                pos_y = 0

                            final_path[pos_y, pos_x] = 1

                    elif beda_x <= 0:
                        for nilai_x in np.arange(cur_x, old_x,0.1):
                            # Persamaan garis lurus
                            nilai_y = (nilai_x*(y2 - y1) - (x1*y2) + (x2*y1))/(x2 - x1)
                            #print("nilai_x = ", nilai_x)
                            #print("nilai_y = ", nilai_y)

                            # Mapping X
                            if -0.5 <= nilai_x <= 0.5:
                                pos_x = 0
                            elif 0.5 < nilai_x <= 1.5:
                                pos_x = 1
                            elif 1.5 < nilai_x <= 2.5:
                                pos_x = 2
                            elif 2.5 < nilai_x <= 3.5:
                                pos_x = 3
                            elif 3.5 < nilai_x <= 4.5:
                                pos_x = 4
                            elif 4.5 < nilai_x <= 5.5:
                                pos_x = 5
                            elif 5.5 < nilai_x <= 6.5:
                                pos_x = 6

                            # Mapping Y
                            if 0 <= nilai_y <= 0.5:
                                pos_y = 5
                            elif 0.5 < nilai_y <= 1.5:
                                pos_y = 4
                            elif 1.5 < nilai_y <= 2.5:
                                pos_y = 3
                            elif 2.5 < nilai_y <= 3.5:
                                pos_y = 2
                            elif 3.5 < nilai_y <= 4.5:
                                pos_y = 1
                            elif 4.5 < nilai_y <= 5.5:
                                pos_y = 0

                            final_path[pos_y, pos_x] = 1

                    # Berdasarkan Y
                    if beda_y > 0:
                        for nilai_y in np.arange(old_y, cur_y,0.1):
                            # Persamaan garis lurus
                            nilai_x = (nilai_y*(x2 - x1) + (x1*y2) - (x2*y1))/(y2 - y1)
                            #print("nilai_x = ", nilai_x)
                            #print("nilai_y = ", nilai_y)

                            # Mapping X
                            if -0.5 <= nilai_x <= 0.5:
                                pos_x = 0
                            elif 0.5 < nilai_x <= 1.5:
                                pos_x = 1
                            elif 1.5 < nilai_x <= 2.5:
                                pos_x = 2
                            elif 2.5 < nilai_x <= 3.5:
                                pos_x = 3
                            elif 3.5 < nilai_x <= 4.5:
                                pos_x = 4
                            elif 4.5 < nilai_x <= 5.5:
                                pos_x = 5
                            elif 5.5 < nilai_x <= 6.5:
                                pos_x = 6

                            # Mapping Y
                            if 0 <= nilai_y <= 0.5:
                                pos_y = 5
                            elif 0.5 < nilai_y <= 1.5:
                                pos_y = 4
                            elif 1.5 < nilai_y <= 2.5:
                                pos_y = 3
                            elif 2.5 < nilai_y <= 3.5:
                                pos_y = 2
                            elif 3.5 < nilai_y <= 4.5:
                                pos_y = 1
                            elif 4.5 < nilai_y <= 5.5:
                                pos_y = 0

                            final_path[pos_y, pos_x] = 1

                    elif beda_y <= 0:
                        for nilai_y in np.arange(cur_y, old_y,0.1):
                            # Persamaan garis lurus
                            nilai_x = (nilai_y*(x2 - x1) + (x1*y2) - (x2*y1))/(y2 - y1)
                            #print("nilai_x = ", nilai_x)
                            #print("nilai_y = ", nilai_y)

                            # Mapping X
                            if -0.5 <= nilai_x <= 0.5:
                                pos_x = 0
                            elif 0.5 < nilai_x <= 1.5:
                                pos_x = 1
                            elif 1.5 < nilai_x <= 2.5:
                                pos_x = 2
                            elif 2.5 < nilai_x <= 3.5:
                                pos_x = 3
                            elif 3.5 < nilai_x <= 4.5:
                                pos_x = 4
                            elif 4.5 < nilai_x <= 5.5:
                                pos_x = 5
                            elif 5.5 < nilai_x <= 6.5:
                                pos_x = 6

                            # Mapping Y
                            if 0 <= nilai_y <= 0.5:
                                pos_y = 5
                            elif 0.5 < nilai_y <= 1.5:
                                pos_y = 4
                            elif 1.5 < nilai_y <= 2.5:
                                pos_y = 3
                            elif 2.5 < nilai_y <= 3.5:
                                pos_y = 2
                            elif 3.5 < nilai_y <= 4.5:
                                pos_y = 1
                            elif 4.5 < nilai_y <= 5.5:
                                pos_y = 0

                            final_path[pos_y, pos_x] = 1


                    old_x = cur_x
                    old_y = cur_y

        print("final_path = \n", final_path)

        fp_file.create_dataset('final_path', data = final_path)
        fp_file.create_dataset('check_pos', data = check_pos)
        fp_file.create_dataset('range_const', data = range_const)
        fp_file.create_dataset('list_save', data = list_save)
        fp_file.create_dataset('tengah', data = tengah)
        
        row = len(final_path)
        col = len(final_path[0])

        single_mask = np.zeros((600, 175, 3), np.uint8)
        final_path_flat = final_path.flatten()

        for i in range(len(final_path_flat)):
            if final_path_flat[i] == 1:
                selected_pos_x = inv_save_x[i]
                selected_pos_y = inv_save_y[i]
                cv2.rectangle(single_mask, (int(selected_pos_x) - range_const, int(selected_pos_y) - 51), (int(selected_pos_x) + range_const, int(selected_pos_y) + 51), (255,0,0), cv2.FILLED)

        cv2.imshow("aa", single_mask)
        cv2.waitKey(0)

        print("")

if __name__ == '__main__':
    data_dir = "/media/ferdyan/LocalDiskE/Hasil/dataset/New/X_ooc14/"
    #main(data_dir)
    pilih(data_dir)
