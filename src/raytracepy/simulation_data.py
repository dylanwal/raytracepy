
import matplotlib.pyplot as plt


class SimData:
    def __init__(self, data_planes, num_traces: int = 100):
        """

        :param data_planes: planes
        :param num_traces:
        """
        if isinstance(data_planes, Plane):
            self.data_planes = [data_planes]
        elif type(data_planes) == list:
            self.data_planes = data_planes
        else:
            exit("Invalid format for planes to be passed into SimData.")

        self.num_traces = num_traces

        # data
        self.num_rays = None
        self.rays_per_light = None
        self.traces_per_light = None
        self.max_bounces = None
        self.planes = None
        self.lights = None
        self.hist = [0] * len(self.data_planes)
        self.traces = []

    def calc_rays_per_light(self, num_lights):
        self.rays_per_light = int(self.num_rays / num_lights)

    def calc_traces_per_light(self, num_lights):
        self.traces_per_light = int(self.num_traces / num_lights)

    def histogram(self, plane, hits):
        x, y, z = np.split(hits, 3, axis=1)
        if plane.normal[0] == 0 and plane.normal[1] == 0:
            his, xedges, yedges = np.histogram2d(np.asarray(x)[:, 0], np.asarray(y)[:, 0], bins=150,
                                                 range=[plane.range[0:2], plane.range[2:4]])
        elif plane.normal[0] == 0 and plane.normal[2] == 0:
            his, xedges, yedges = np.histogram2d(np.asarray(x)[:, 0], np.asarray(z)[:, 0], bins=150,
                                                 range=[plane.range[0:2], plane.range[2:4]])
        else:
            his, xedges, yedges = np.histogram2d(np.asarray(y)[:, 0], np.asarray(z)[:, 0], bins=150,
                                                 range=[plane.range[0:2], plane.range[2:4]])

        try:
            if self.hist[plane.id] == 0:
                self.hist[plane.id] = his
        except ValueError:
            self.hist[plane.id] = np.add(self.hist[plane.id], his)

    def plot_hist(self):
        for plane, hist in zip(self.planes, self.hist):
            plotting.plot_hist(hist, plane)

    def plot_traces(self):
        plotting.plot_traces(self.planes, self.lights, self.traces)

    def hit_stats(self):
        print("\n")
        print(f"rays generated: {self.num_rays}")
        print(f"rays per light: {self.rays_per_light}")
        print(f"max bounces: {self.max_bounces}")
        print(f"number of lights in simulation: {len(self.lights)}")
        print(f"number of planes in simulation: {len(self.planes)}")
        for hist, plane in zip(self.hist, self.data_planes):
            print(f"Plane: \tid: {plane.id} \t type: {plane.trans_type}")
            print(f"rays hit surface: {np.sum(hist)}, ({np.sum(hist)/self.num_rays})")

    def percentile_table(self, normalized=False):
        print("\n")
        headers = ["min", "1", "5", "10", "mean", "90", "95", "99", "max"]
        if normalized:
            for hist, plane in zip(self.hist, self.data_planes):
                print(f"Plane: \tid: {plane.id} \t type: {plane.trans_type}")
                his_array = np.reshape(hist, (hist.shape[0] * hist.shape[1],))
                mean = np.mean(his_array)
                data = [sig_figs(np.min(his_array)),
                        sig_figs(np.min(his_array)/mean),
                        sig_figs(np.percentile(his_array, 1)/mean),
                        sig_figs(np.percentile(his_array, 5)/mean),
                        sig_figs(np.percentile(his_array, 10)/mean),
                        sig_figs(mean/mean),
                        sig_figs(np.percentile(his_array, 90)/mean),
                        sig_figs(np.percentile(his_array, 95)/mean),
                        sig_figs(np.percentile(his_array, 99)/mean),
                        sig_figs(np.max(his_array)/mean)]

                print(tabulate([data], headers=headers))
        else:
            for hist, plane in zip(self.hist, self.data_planes):
                print(f"Plane: \tid: {plane.id} \t type: {plane.trans_type}")
                his_array = np.reshape(hist, (hist.shape[0] * hist.shape[1],))
                mean = np.mean(his_array)
                data = [sig_figs(np.min(his_array)),
                        sig_figs(np.min(his_array)),
                        sig_figs(np.percentile(his_array, 1)),
                        sig_figs(np.percentile(his_array, 5)),
                        sig_figs(np.percentile(his_array, 10)),
                        sig_figs(mean),
                        sig_figs(np.percentile(his_array, 90)),
                        sig_figs(np.percentile(his_array, 95)),
                        sig_figs(np.percentile(his_array, 99)),
                        sig_figs(np.max(his_array))]

                print(tabulate([data], headers=headers))

    def save_data(self, file_name: str = "data"):
        with open(file_name + '.pickle', 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load_data(file_name: str = "data"):
        with open(file_name + '.pickle', 'rb') as file2:
            return pickle.load(file2)

    def plotting_light_positions(positions):
        """
        Takes x,y and generates a simple plot of light locations.
        :param positions:
        :return: plot
        """
        x, y = np.split(positions, 2, axis=1)
        plt.plot(x, y, 'ko')
        plt.show()

    def plotting_lens_angles(x, y):
        """
        Takes x,y and generates a simple plot
        :param x:
        :param y:
        :return:
        """
        plt.plot(x, y, 'k-')
        plt.show()

    def plane_corner_xyz(plane):
        if plane.normal[2] != 0:
            d = - np.dot(plane.position, plane.normal)
            xx, yy = np.meshgrid([plane.corners[0], plane.corners[1]], [plane.corners[2], plane.corners[3]])
            zz = (-plane.normal[0] * xx - plane.normal[1] * yy - d) * 1. / plane.normal[2]
        else:
            if plane.normal[0] == 0:
                xx = np.array([[plane.corners[0], plane.corners[0]], [plane.corners[1], plane.corners[1]]])
                yy = np.array([[plane.corners[2], plane.corners[3]], [plane.corners[2], plane.corners[3]]])
                zz = np.array([[plane.corners[4], plane.corners[5]], [plane.corners[4], plane.corners[5]]])
            if plane.normal[1] == 0:
                xx = np.array([[plane.corners[0], plane.corners[0]], [plane.corners[1], plane.corners[1]]])
                yy = np.array([[plane.corners[2], plane.corners[3]], [plane.corners[2], plane.corners[3]]])
                zz = np.array([[plane.corners[4], plane.corners[4]], [plane.corners[5], plane.corners[5]]])

        return xx, yy, zz

    def plot_traces(planes: list, lights: list, traces: list):
        ax = plt.axes(projection='3d')

        # Planes
        for plane in planes:
            alpha = 0.6
            color = (0.5, 0.5, 0.5)
            if plane.trans_type == 0:
                alpha = 1
                color = (0, 0, 1)
            elif plane.trans_type == 1:
                alpha = 0.2
                color = (0, 0, 0)

            xx, yy, zz = plane_corner_xyz(plane)
            ax.plot_surface(xx, yy, zz, color=color, alpha=alpha)

        # Lights
        for light in lights:
            ax.scatter3D(light.position[0], light.position[1], light.position[2], s=40, color=(0, 0.8, 0.8))
            ax.quiver(light.position[0], light.position[1], light.position[2],
                      light.direction[0], light.direction[1], light.direction[2],
                      length=4, color=(0.1, 0.2, 0.5))

        # Traces
        if not traces == []:
            for trace in traces:
                x = trace[:, 0]
                y = trace[:, 1]
                z = trace[:, 2]
                ax.plot(x, y, z)
                ax.scatter3D(x[-1], y[-1], z[-1], color=(1, 0, 0), alpha=1, s=4)

        # ax.set_xlim3d(-10, 1)
        # ax.set_ylim3d(-1, 1)
        # ax.set_zlim3d(-1, 1)
        plt.show()

    def plot_hist(his, plane):
        plt.imshow(his, origin='lower', aspect='auto', extent=plane.range)
        cb = plt.colorbar()
        cb.set_label("density")
        plt.clim(0, np.percentile(np.reshape(his, (his.shape[0] * his.shape[1],)), 95))
        plt.show()