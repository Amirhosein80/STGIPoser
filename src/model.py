from collections import OrderedDict

from src.base_model import BaseModel, run_benchmark
from src.blocks import *
from src.config_reader import read_config
from src.logger import get_logger
from src.smpl_model import ParametricModel

logger = get_logger(__name__)


class STIPoser(BaseModel):
    """A Spatial-Temporal IMU-based Pose Estimation model (STIPoser).

    This model processes IMU data to estimate human pose, optionally incorporating UWB data
    and translation information. It uses a combination of GRU and GCN layers for temporal
    and spatial processing of IMU signals.

    Parameters
    ----------
    configs : dict
        Configuration dictionary containing model parameters including:
        - embed_dim: Dimension of embedding layers
        - use_uwb: Whether to use UWB data
        - use_translation: Whether to estimate translation
        - use_imu_aux: Whether to use auxiliary IMU predictions
    smpl_model : ParametricModel
        SMPL model instance for human body modeling

    Attributes
    ----------
    embed_dim : int
        Dimension of embedding layers
    use_uwb : bool
        Flag for using UWB data
    use_translation : bool
        Flag for translation estimation
    use_imu_aux : bool
        Flag for auxiliary IMU predictions
    state_inited : bool
        Whether states have been initialized
    states : list
        List of hidden states for recurrent layers
    """

    def __init__(self, configs: dict, smpl_model: ParametricModel) -> None:
        """Initialize the STIPoser model.

        Parameters
        ----------
        configs : dict
            Configuration dictionary
        smpl_model : ParametricModel
            SMPL model instance
        """
        super().__init__(smpl_model=smpl_model, configs=configs)

        self.embed_dim = configs["model"]["embed_dim"]
        self.use_uwb = configs["model"]["use_uwb"]
        self.use_translation = configs["model"]["use_translation"]
        self.use_imu_aux = configs["model"]["use_imu_aux"]
        self.use_uwb_attn = configs["model"]["use_uwb_attn"]

        logger.info(
            f"Initialize STIPoser model with embed_dim: {self.embed_dim},"
            f"use_uwb: {self.use_uwb}, use_translation: {self.use_translation},"
            f"use_imu_aux: {self.use_imu_aux}, use_uwb_attn: {self.use_uwb_attn}"
        )

        self.temp1_gru = MultiGRU(self.embed_dim, self.embed_dim)
        self.temp2_gru = MultiGRU(self.embed_dim, self.embed_dim)

        self.imus_spatial1 = GCNBlock(
            self.embed_dim,
            self.embed_dim,
            (self.adj_imu, self.adj_imu),
            self.use_uwb_attn,
        )
        self.imus_spatial2 = GCNBlock(
            self.embed_dim,
            self.embed_dim,
            (self.adj_imu, self.adj_imu),
            self.use_uwb_attn,
        )
        self.pose_spatial = GCNLayer(self.embed_dim, self.embed_dim, self.adj_imu2parts)

        self.vel_init = nn.Sequential(
            GCNLayer(3, self.embed_dim * 2, self.adj_imu),
            G_Act(),
        )

        self.imu_embedding = nn.Sequential(
            nn.Linear(18 if self.use_uwb else 12, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.ReLU(),
        )

        self.pose_head_foot = nn.Sequential(
            nn.LayerNorm(self.embed_dim), nn.ReLU(), nn.Linear(self.embed_dim, 6 * 2)
        )

        self.pose_head_body = nn.Sequential(
            nn.LayerNorm(self.embed_dim), nn.ReLU(), nn.Linear(self.embed_dim, 6 * 4)
        )

        self.pose_head_hand = nn.Sequential(
            nn.LayerNorm(self.embed_dim), nn.ReLU(), nn.Linear(self.embed_dim, 6 * 4)
        )

        if self.use_translation:
            self.tran_head = MultiGRU(self.embed_dim, 3, shortcut=False)
            self.tran_init = nn.Sequential(
                nn.Linear(3, self.embed_dim),
                nn.LayerNorm(self.embed_dim),
                nn.ReLU(),
                nn.Linear(self.embed_dim, self.embed_dim * 2),
                G_Act(),
            )

        if self.use_imu_aux:
            self.imuv_head = nn.Linear(self.embed_dim, 3)

        self.state_inited = False
        self.states = [None, None, None, None]

    def forward(self, datas: dict) -> OrderedDict:
        """Forward pass of the STIPoser model for model training.

        Parameters
        ----------
        datas : dict
            Dictionary containing input data including:
            - imu_acc: IMU acceleration
            - imu_ori: IMU orientation
            - uwb: UWB data
            - last_jvel: Last joint velocity
            - last_trans: Last translation

        Returns
        -------
        OrderedDict
            Dictionary containing model outputs including:
            - out_rot: Rotation predictions
            - out_trans: Translation predictions if self.use_translation is True
            - imus_velocity: IMU velocity predictions if self.use_imu_aux is True

        Examples
        --------
        >>> datas = {
            "imu_acc": torch.randn(32, 300, 6, 3),
            "imu_ori": torch.randn(32, 300, 6, 3, 3),
            "uwb": torch.randn(32, 300, 6, 6),
            "last_jvel": torch.randn(32, 6, 3),
            "last_trans": torch.randn(32, 3),
        }
        >>> outputs = model(datas)
        """
        outputs = OrderedDict()

        imu_acc = datas["imu_acc"]
        imu_ori = datas["imu_ori"]
        uwb = datas["uwb"]

        x, _ = self.normalize(imu_acc=imu_acc, imu_ori=imu_ori, casual_state=None)

        batch_size, seq_len, num_imus, _ = imu_acc.size()
        last_vimu = datas["last_jvel"].to(x.dtype)

        hs1 = self.vel_init(last_vimu[:, self.leaf].unsqueeze(1))
        hs1 = list(hs1.chunk(2, dim=-1))

        if self.use_uwb:
            x = torch.cat([x, uwb], dim=-1)
        else:
            uwb = None

        x = self.imu_embedding(x)
        x = self.imus_spatial1(x, uwb)
        x, _ = self.temp1_gru(x, hs1)
        x = self.imus_spatial2(x, uwb)
        x, _ = self.temp2_gru(x, None)
        if self.use_imu_aux:
            outputs["aux_ivel"] = self.imuv_head(x).reshape(batch_size, seq_len, 6, 3)

        x = self.pose_spatial(x)

        if self.use_translation:
            last_trans = datas["last_trans"].to(x.dtype)
            hs3 = self.tran_init(last_trans).unsqueeze(1).unsqueeze(1)
            hs3 = list(hs3.chunk(2, dim=-1))

            t, _ = self.tran_head(x[:, :, -1:], hs3)
            outputs["out_trans"] = t.squeeze(-2)

        x = torch.cat(
            [
                self.pose_head_foot(x[:, :, 0:1]).reshape(batch_size, seq_len, 2, 6),
                self.pose_head_body(x[:, :, 1:2]).reshape(batch_size, seq_len, 4, 6),
                self.pose_head_hand(x[:, :, 2:3]).reshape(batch_size, seq_len, 4, 6),
            ],
            dim=-2,
        )

        outputs["out_rot"] = x

        return outputs

    def init_state(self):
        """Initialize the states of the model.

        This method initializes the states of the model for the online inference.

        Returns
        -------
        None
        """
        self.states = [None, None, None, None]

        last_vimu = torch.zeros((1, 1, 6, 3), device=self.device)

        hs1 = self.vel_init(last_vimu)
        hs1 = list(hs1.chunk(2, dim=-1))
        self.states[1] = hs1

        if self.use_translation:
            last_trans = torch.zeros((1, 1, 1, 3), device=self.device)
            hs3 = self.tran_init(last_trans)
            hs3 = list(hs3.chunk(2, dim=-1))
            self.states[3] = hs3

        self.state_inited = True

    @torch.no_grad()
    def forward_online(self, imu_acc, imu_ori, uwb):
        """Forward pass of the STIPoser model for online inference.

        This method performs a forward pass of the STIPoser model for online inference.

        Parameters
        ----------
        imu_acc : torch.Tensor
            IMU acceleration
        imu_ori : torch.Tensor
            IMU orientation
        uwb : torch.Tensor
            UWB data

        Returns
        -------
        sensors : torch.Tensor
            IMU orientation
        t : torch.Tensor
            Translation predictions

        Examples
        --------
        >>> imu_acc = torch.randn(6, 3)
        >>> imu_ori = torch.randn(6, 3, 3)
        >>> uwb = torch.randn(6, 6)
        >>> ori, t = model.forward_online(imu_acc, imu_ori, uwb)
        """
        sensors = imu_ori.clone().unsqueeze(0)

        if not self.state_inited:
            self.init_state()

        imu_acc = imu_acc.unsqueeze(0).unsqueeze(0)
        imu_ori = imu_ori.unsqueeze(0).unsqueeze(0)

        x, self.states[0] = self.normalize(
            imu_acc=imu_acc, imu_ori=imu_ori, casual_state=self.states[0]
        )

        if self.use_uwb:
            uwb = uwb.unsqueeze(0).unsqueeze(0)
            x = torch.cat([x, uwb], dim=-1)
        else:
            uwb = None

        x = self.imu_embedding(x)

        x = self.imus_spatial1(x, uwb)
        x, self.states[1] = self.temp1_gru(x, self.states[1])
        x = self.imus_spatial2(x, uwb)
        x, self.states[2] = self.temp2_gru(x, self.states[2])

        x = self.pose_spatial(x)

        if self.use_translation:
            t, self.states[3] = self.tran_head(x[:, :, -1:], self.states[3])
            t = t.reshape(1, 3)

        else:
            t = None

        x = torch.cat(
            [
                self.pose_head_foot(x[:, :, 0:1]).reshape(1, 2, 6),
                self.pose_head_body(x[:, :, 1:2]).reshape(1, 4, 6),
                self.pose_head_hand(x[:, :, 2:3]).reshape(1, 4, 6),
            ],
            dim=-2,
        )

        return self.glb_6d_to_full_local_mat(x, sensor_rot=sensors), t

    @torch.no_grad()
    def forward_offline(self, imu_acc, imu_ori, uwb):
        """Forward pass of the STIPoser model for offline inference.

        This method performs a forward pass of the STIPoser model for offline inference.

        Parameters
        ----------
        imu_acc : torch.Tensor
            IMU acceleration
        imu_ori : torch.Tensor
            IMU orientation
        uwb : torch.Tensor
            UWB data
       
        Returns
        -------
        x : torch.Tensor
            6D rotation predictions or full local rotation matrix predictions
        t : torch.Tensor
            Translation predictions

        Examples
        --------
        >>> imu_acc = torch.randn(300, 6, 3)
        >>> imu_ori = torch.randn(300, 6, 3, 3)
        >>> uwb = torch.randn(300, 6, 6)
        >>> ori, t = model.forward_offline(imu_acc, imu_ori, uwb) # ori is full local rotation matrix
        """
        sensors = imu_ori.clone()

        T = imu_acc.shape[0]

        imu_acc = imu_acc.unsqueeze(0)
        imu_ori = imu_ori.unsqueeze(0)

        x, _ = self.normalize(imu_acc=imu_acc, imu_ori=imu_ori, casual_state=None)

        if self.use_uwb:
            uwb = uwb.unsqueeze(0)
            x = torch.cat([x, uwb], dim=-1)
        else:
            uwb = None

        last_vimu = torch.zeros((1, 1, 6, 3), device=self.device)
        hs1 = self.vel_init(last_vimu)
        hs1 = list(hs1.chunk(2, dim=-1))

        x = self.imu_embedding(x)

        x = self.imus_spatial1(x, uwb)
        x, _ = self.temp1_gru(x, hs1)
        x = self.imus_spatial2(x, uwb)
        x, _ = self.temp2_gru(x, None)
        x = self.pose_spatial(x)

        if self.use_translation:
            last_trans = torch.zeros((1, 1, 1, 3), device=self.device)
            hs3 = self.tran_init(last_trans)
            hs3 = list(hs3.chunk(2, dim=-1))
            t, _ = self.tran_head(x[:, :, -1:], hs3)
            t = t.reshape(T, 3)
        else:
            t = None

        x = torch.cat(
            [
                self.pose_head_foot(x[:, :, 0:1]).reshape(T, 2, 6),
                self.pose_head_body(x[:, :, 1:2]).reshape(T, 4, 6),
                self.pose_head_hand(x[:, :, 2:3]).reshape(T, 4, 6),
            ],
            dim=-2,
        )

        return self.glb_6d_to_full_local_mat(x, sensor_rot=sensors), t


if __name__ == "__main__":
    import torchinfo

    torch.set_float32_matmul_precision("high")

    configs = read_config("./config/config.yaml")
    configs["training"]["device"] = "cpu"
    smpl_model = ParametricModel(configs["smpl"]["file"])
    model = STIPoser(configs, smpl_model)
    device = configs["training"]["device"]

    model.forward = model.forward_offline
    torchinfo.summary(
        model=model,
        input_data=[
            torch.randn(3600, 6, 3).to(device),
            torch.randn(3600, 6, 3, 3).to(device),
            torch.randn(3600, 6, 6).to(device),
        ],
        device=device,
        verbose=1,
    )

    model.forward = model.forward_online
    torchinfo.summary(
        model=model,
        input_data=[
            torch.randn(6, 3).to(device),
            torch.randn(6, 3, 3).to(device),
            torch.randn(6, 6).to(device),
        ],
        device=device,
        verbose=1,
    )
    run_benchmark(
        model,
        datas=(
            torch.randn(6, 3).to(device),
            torch.randn(6, 3, 3).to(device),
            torch.randn(6, 6).to(device),
        ),
    )
