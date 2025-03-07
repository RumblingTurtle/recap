import os
import glob
import joblib
import torch
import numpy as np

from recap.amass.motion_wrapper import AMASSMotionWrapper
from recap.amass.transforms import calculate_body_transforms


class AMASSMotionLoader:
    def __init__(
        self,
        datasets_path,
        motion_filename=None,
        target_fps=60,
        template_scale=1.0,
        beta=None,
        shuffle=False,
        name_blacklist=None,
        name_whitelist=None,
        device="cpu",
    ):
        self.datasets_path = datasets_path
        self.device = device
        self.target_fps = target_fps
        self.template_scale = torch.tensor(template_scale, device=device)
        if motion_filename is None:
            self.paths = glob.glob(os.path.join(datasets_path, "amass/datasets/**/*.npz"), recursive=True)
            self.paths = self.filter_paths(name_blacklist, name_whitelist)
            if shuffle:
                np.random.shuffle(self.paths)
        else:
            self.paths = glob.glob(
                os.path.join(self.datasets_path, "amass/datasets/**/", motion_filename),
                recursive=True,
            )
            if len(self.paths) == 0:
                raise ValueError(f"No motion file found for {motion_filename}")
            if len(self.paths) > 1:
                print(
                    f"Multiple motion files found for {motion_filename}. Using the first one {os.linesep}\
                    {os.linesep.join(self.paths)}"
                )
            self.paths = [self.paths[0]]
        self.path_iterator = iter(self.paths)
        self.occlusion = joblib.load(os.path.join(self.datasets_path, "amass_copycat_occlusion_v3.pkl"))
        # Load body templates
        self.body_templates = {}
        self.dmpl_templates = {}
        for gender in ["neutral", "male", "female"]:
            bm_fname = os.path.join(self.datasets_path, f"amass/body_models/smplh/{gender}/model.npz")
            dmpl_fname = os.path.join(self.datasets_path, f"amass/body_models/dmpls/{gender}/model.npz")

            self.body_templates[gender] = {}
            self.dmpl_templates[gender] = {}
            body_template = np.load(bm_fname)
            dmpl_template = np.load(dmpl_fname)

            for key in [
                "J_regressor",
                "shapedirs",
                "v_template",
                "kintree_table",
            ]:
                value = torch.from_numpy(body_template[key]).to(self.device)
                value.requires_grad = False
                self.body_templates[gender][key] = value
            eigvec = torch.from_numpy(dmpl_template["eigvec"]).to(self.device)
            eigvec.requires_grad = False
            self.dmpl_templates[gender]["eigvec"] = eigvec

        self.beta = torch.from_numpy(beta).to(self.device) if beta is not None else None

    def filter_paths(self, name_blacklist, name_whitelist):
        filtered_paths = []
        for path in self.paths:
            basename = os.path.basename(path).lower()

            if name_blacklist is not None:
                keep_clip = True
                for skippable_name in name_blacklist:
                    if skippable_name in basename:
                        keep_clip = False
                        break

                if not keep_clip:
                    continue

            if name_whitelist is not None:
                keep_clip = False
                for keepable_name in name_whitelist:
                    if keepable_name in basename:
                        keep_clip = True
                        break

                if keep_clip:
                    filtered_paths.append(path)
            else:
                filtered_paths.append(path)
        return filtered_paths

    def __iter__(
        self,
    ):
        return self

    def __next__(
        self,
    ):
        self.current_path = next(self.path_iterator)
        while "shape.npz" in self.current_path:
            self.current_path = next(self.path_iterator)
        return self.preprocess(self.current_path)

    def __len__(self):
        return len(self.paths)

    def get_transform_args(self):
        clip_data = self._load_and_validate_clip(self.current_path)
        if clip_data is None:
            return None

        _, poses, root_pos, blendshapes, _, gender = clip_data

        return {
            "poses": poses,
            "root_pos": root_pos,
            "blendshapes": blendshapes,
            "shapedirs": self.body_templates[gender]["shapedirs"],
            "eigvec": self.dmpl_templates[gender]["eigvec"],
            "v_template": self.body_templates[gender]["v_template"],
            "J_regressor": self.body_templates[gender]["J_regressor"],
            "kintree_table": self.body_templates[gender]["kintree_table"],
        }

    def preprocess(self, path) -> AMASSMotionWrapper:
        clip_data = self._load_and_validate_clip(path)
        if clip_data is None:
            return None

        clip_name, poses, root_pos, blendshapes, betas, gender = clip_data
        with torch.no_grad():
            positions, rotations = calculate_body_transforms(
                poses=poses,
                root_pos=root_pos,
                betas=betas,
                blendshapes=blendshapes,
                shapedirs=self.body_templates[gender]["shapedirs"],
                eigvec=self.dmpl_templates[gender]["eigvec"],
                v_template=self.body_templates[gender]["v_template"],
                J_regressor=self.body_templates[gender]["J_regressor"],
                kintree_table=self.body_templates[gender]["kintree_table"],
                template_scale=self.template_scale,
            )
            return clip_name, AMASSMotionWrapper(positions=positions, rotations=rotations)

    def _load_and_validate_clip(self, path):
        """Load and validate clip data, handling frame rate and sequence issues"""
        clip_dict = np.load(path)
        clip_name = os.path.basename(path).split(".")[0]

        gender = clip_dict["gender"]
        if isinstance(gender, np.ndarray):
            gender = gender.item()
        if isinstance(gender, bytes):
            gender = gender.decode("utf-8")

        clip_framerate = clip_dict["mocap_framerate"]
        skip = int(clip_framerate / self.target_fps)

        poses = torch.from_numpy(clip_dict["poses"][::skip]).to(self.device)
        root_pos = torch.from_numpy(clip_dict["trans"][::skip]).to(self.device)
        blendshapes = torch.from_numpy(clip_dict["dmpls"][::skip]).to(self.device)
        seq_length = poses.shape[0]

        # Handle occlusion and sequence length validation
        if clip_name in self.occlusion:
            issue = self.occlusion[clip_name]["issue"]
            if (issue == "sitting" or issue == "airborne") and "idxes" in self.occlusion[clip_name]:
                return self.occlusion[clip_name]["idxes"][0]
            else:
                print("issue irrecoverable", clip_name, issue)
                return None

        if seq_length is None or seq_length < 10:
            return None

        # Trim sequences to valid length
        poses = poses[:seq_length]
        root_pos = root_pos[:seq_length]
        blendshapes = blendshapes[:seq_length]
        beta = self.beta if self.beta is not None else torch.from_numpy(clip_dict["betas"]).to(self.device)

        return clip_name, poses, root_pos, blendshapes, beta, gender
