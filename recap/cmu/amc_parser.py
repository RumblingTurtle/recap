import numpy as np

"""
Original work by Yuxiao Zhou 
https://github.com/CalciferZh/AMCParser
"""

"""
http://mocap.cs.cmu.edu/faqs.php
How do I convert the lengths used in the asf/amc files into meters?
The bone length and the root position data for asf/amc files that are in the database should be multiplied by this scale to convert to meters:

(1.0/0.45)*2.54/100.0 = 0.056444

0.45 is the scale from ASF file (define in the header)
2.54/100 to convert from inches to meters because the data is in inches
"""
ASF_TO_METERS = 0.056444


def euler2mat(theta):
    R = np.array(
        [
            [
                np.cos(theta[1]) * np.cos(theta[2]),
                np.sin(theta[0]) * np.sin(theta[1]) * np.cos(theta[2]) - np.sin(theta[2]) * np.cos(theta[0]),
                np.sin(theta[1]) * np.cos(theta[0]) * np.cos(theta[2]) + np.sin(theta[0]) * np.sin(theta[2]),
            ],
            [
                np.sin(theta[2]) * np.cos(theta[1]),
                np.sin(theta[0]) * np.sin(theta[1]) * np.sin(theta[2]) + np.cos(theta[0]) * np.cos(theta[2]),
                np.sin(theta[1]) * np.sin(theta[2]) * np.cos(theta[0]) - np.sin(theta[0]) * np.cos(theta[2]),
            ],
            [
                -np.sin(theta[1]),
                np.sin(theta[0]) * np.cos(theta[1]),
                np.cos(theta[0]) * np.cos(theta[1]),
            ],
        ]
    )

    return R


class Joint:
    def __init__(self, name, direction, length, axis, dof, limits):
        """
        Definition of basic joint. The joint also contains the information of the
        bone between it's parent joint and itself. Refer
        [here](https://research.cs.wisc.edu/graphics/Courses/cs-838-1999/Jeff/ASF-AMC.html)
        for detailed description for asf files.

        Parameter
        ---------
        name: Name of the joint defined in the asf file. There should always be one
        root joint. String.

        direction: Default direction of the joint(bone). The motions are all defined
        based on this default pose.

        length: Length of the bone.

        axis: Axis of rotation for the bone.

        dof: Degree of freedom. Specifies the number of motion channels and in what
        order they appear in the AMC file.

        limits: Limits on each of the channels in the dof specification

        """
        self.name = name
        self.direction = np.reshape(direction, [3, 1])
        self.length = length
        axis = np.deg2rad(axis)
        self.C = euler2mat(axis)
        self.Cinv = np.linalg.inv(self.C)
        self.limits = np.zeros([3, 2])
        for lm, nm in zip(limits, dof):
            if nm == "rx":
                self.limits[0] = lm
            elif nm == "ry":
                self.limits[1] = lm
            else:
                self.limits[2] = lm
        self.parent = None
        self.children = []
        self.coordinate = None
        self.matrix = None
        self.R_to_rhs = np.array(
            [
                [0, 0, 1],
                [1, 0, 0],
                [0, 1, 0],
            ],
            np.float32,
        )
        self.height_offset = 0

    def set_motion(self, motion=None):
        if motion is None or self.name not in motion.keys():
            motion_value = np.zeros(6)
        else:
            motion_value = motion[self.name]

        if self.name == "root":
            self.coordinate = np.array(motion_value[:3])[:, None] * ASF_TO_METERS
            self.rotation = np.deg2rad(motion_value[3:])
            self.matrix = self.C.dot(euler2mat(self.rotation)).dot(self.Cinv)
        else:
            idx = 0
            rotation = np.zeros(3)
            for axis, lm in enumerate(self.limits):
                if not np.array_equal(lm, np.zeros(2)):
                    rotation[axis] = motion_value[idx]
                    idx += 1
            self.rotation = np.deg2rad(rotation)
            self.matrix = self.parent.matrix.dot(self.C).dot(euler2mat(self.rotation)).dot(self.Cinv)
            self.coordinate = self.parent.coordinate + self.length * self.matrix.dot(self.direction)
        for child in self.children:
            child.set_motion(motion)

    def to_dict(self):
        ret = {self.name: self}
        for child in self.children:
            ret.update(child.to_dict())
        return ret

    def pretty_print(self):
        print("===================================")
        print("joint: %s" % self.name)
        print("direction:")
        print(self.direction)
        print("limits:", self.limits)
        print("parent:", self.parent)
        print("children:", self.children)

    def set_zero(
        self,
    ):
        self.set_motion()

    @property
    def position(self):
        return self.R_to_rhs @ self.coordinate.squeeze()

    @property
    def R(self):
        return (self.R_to_rhs @ self.matrix)[:, [2, 0, 1]]


def read_line(stream, idx):
    if idx >= len(stream):
        return None, idx
    line = stream[idx].strip().split()
    idx += 1
    return line, idx


def parse_asf(file_path, reassign_lengths={}):
    """read joint data only"""
    with open(file_path) as f:
        content = f.read().splitlines()

    for idx, line in enumerate(content):
        # meta infomation is ignored
        if line == ":bonedata":
            content = content[idx + 1 :]
            break

    # read joints
    joints = {"root": Joint("root", np.zeros(3), 0, np.zeros(3), [], [])}
    idx = 0
    while True:
        # the order of each section is hard-coded

        line, idx = read_line(content, idx)

        if line[0] == ":hierarchy":
            break

        assert line[0] == "begin"

        line, idx = read_line(content, idx)
        assert line[0] == "id"

        line, idx = read_line(content, idx)
        assert line[0] == "name"
        name = line[1]

        line, idx = read_line(content, idx)
        assert line[0] == "direction"
        direction = np.array([float(axis) for axis in line[1:]])

        # skip length
        line, idx = read_line(content, idx)
        assert line[0] == "length"
        if name in reassign_lengths.keys():
            length = reassign_lengths[name]
        else:
            length = float(line[1]) * ASF_TO_METERS

        line, idx = read_line(content, idx)
        assert line[0] == "axis"
        assert line[4] == "XYZ"

        axis = np.array([float(axis) for axis in line[1:-1]])

        dof = []
        limits = []

        line, idx = read_line(content, idx)
        if line[0] == "dof":
            dof = line[1:]
            for i in range(len(dof)):
                line, idx = read_line(content, idx)
                if i == 0:
                    assert line[0] == "limits"
                    line = line[1:]
                assert len(line) == 2
                mini = float(line[0][1:])
                maxi = float(line[1][:-1])
                limits.append((mini, maxi))

            line, idx = read_line(content, idx)

        assert line[0] == "end"
        joints[name] = Joint(name, direction, length, axis, dof, limits)

    # read hierarchy
    assert line[0] == ":hierarchy"

    line, idx = read_line(content, idx)

    assert line[0] == "begin"

    while True:
        line, idx = read_line(content, idx)
        if line[0] == "end":
            break
        assert len(line) >= 2
        for joint_name in line[1:]:
            joints[line[0]].children.append(joints[joint_name])
        for nm in line[1:]:
            joints[nm].parent = joints[line[0]]

    return joints


def parse_amc(file_path):
    with open(file_path) as f:
        content = f.read().splitlines()

    for idx, line in enumerate(content):
        if line == ":DEGREES":
            content = content[idx + 1 :]
            break

    frames = []
    idx = 0
    line, idx = read_line(content, idx)
    assert line[0].isnumeric(), line
    EOF = False
    while not EOF:
        joint_degree = {}
        while True:
            line, idx = read_line(content, idx)
            if line is None:
                EOF = True
                break
            if line[0].isnumeric():
                break
            joint_degree[line[0]] = [float(deg) for deg in line[1:]]
        frames.append(joint_degree)
    return frames
