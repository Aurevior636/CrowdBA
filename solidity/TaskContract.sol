// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract TaskContract {
    address public creator;
    string public ipfsHash; // 全局状态变量

    // SubmitData.sol 中的变量
    address public owner;
    uint256 public maxWorkers;
    uint256 public timeEnd1;
    event HashSubmitted(address indexed worker, bytes32 submittedHash);
    mapping(address => bytes32) public submissions;
    mapping(address => bool) public eligibleWorkers;
    bytes32[] public allSubmissions;
    uint256 public currentEligibleWorkers;
    uint256 public submittedWorkers;

    // RewardDistribution.sol 中的变量
    uint256 public rewardCalculateDeadline;
    uint256 public maxRewardCalculateSubmissions;
    mapping(uint256 => Task) private taskData;
    struct Task {
        uint256 task;
        mapping(bytes32 => uint256) votes;
        mapping(address => bytes32) submitters;
        bytes32[] dataHashes;
        uint256 submissionCount;
        bytes32 currentMostVotedHash; // 当前投票数最多的哈希
        uint256 currentHighestVotes;  // 当前投票数最多的哈希对应的投票数
    }
    event TaskSubmitted(address indexed worker, uint256 task, bytes32 dataHash);
    event WorkerReward(address indexed worker, uint256 reward);
    // 事件：当有新的任务申请时触发
    event TaskApplied(address indexed applicant, bytes encryptedData, uint256 requestedAmount);

    // 新增保证金变量
    uint256 public depositAmount;  // 存储任务的保证金数量
    mapping(address => uint256) public deposits;  // 存储每个地址的保证金余额

    // 存储任务申请的加密数据
    struct TaskRequest {
        bytes encryptedData;  // 存储加密的数据
        uint256 requestedAmount;  // 用户请求的标注数量
        address applicant;  // 申请者地址
    }
    TaskRequest[] public taskRequests;

    bool private initialized;

    function initialize(
        address _creator,
        uint256 _SubmitDataDURATION,
        uint256 _maxSubmissions,
        uint256 _rewardDuration,
        uint256 _rewardMaxSubmissions,
        string memory _ipfsHash,
        uint256 _depositAmount  // 新增保证金参数
    ) public {
        require(!initialized, "Contract instance has already been initialized");
        initialized = true;

        creator = _creator;
        ipfsHash = _ipfsHash; // 初始化状态变量

        // 初始化 SubmitData.sol 中的变量
        owner = _creator;
        timeEnd1 = block.timestamp + _SubmitDataDURATION;
        maxWorkers = _maxSubmissions;

        // 初始化 RewardDistribution.sol 中的变量
        rewardCalculateDeadline = block.timestamp + _rewardDuration;
        maxRewardCalculateSubmissions = _rewardMaxSubmissions;

        // 设置保证金金额
        depositAmount = _depositAmount;
    }

    function getIpfsHash() public view returns (string memory) {
        return ipfsHash;
    }

    // SubmitData.sol 中的函数
    function addEligibleWorker(address workerAddress) public {
        require(msg.sender == owner, "Only the owner can call this function");
        require(currentEligibleWorkers < maxWorkers, "Max workers reached");
        eligibleWorkers[workerAddress] = true;
        currentEligibleWorkers++;
    }

    // 修改函数的参数名，避免与状态变量冲突
    function submitHash(bytes32 submittedIpfsHash) public payable {
        require(eligibleWorkers[msg.sender], "You are not an eligible worker");
        require(msg.value == depositAmount, "You must submit the exact required deposit amount");

        // 记录用户提交的哈希
        submissions[msg.sender] = submittedIpfsHash;
        submittedWorkers++;
        emit HashSubmitted(msg.sender, submittedIpfsHash);

        // 处理存入的保证金
        deposits[msg.sender] += msg.value;  // 记录用户支付的保证金
    }

    function getSubmission(address worker) public view returns (bytes32) {
        return submissions[worker];
    }

    function submitCID(string memory ipfsHashStr) public returns (bool) {
        require(submissions[msg.sender] != bytes32(0), "You have not submitted any data");
        require(block.timestamp >= timeEnd1 || submittedWorkers >= maxWorkers, "Cannot call function before deadline or max submissions reached");

        bytes32 ipfsHashBytes = keccak256(abi.encodePacked(ipfsHashStr));
        
        if (ipfsHashBytes == submissions[msg.sender]) {
            allSubmissions.push(ipfsHashBytes);
            emit HashSubmitted(msg.sender, ipfsHashBytes);
            return true;
        }
        return false;
    }

    function getAllSubmissions() public view returns (string[] memory) {
        string[] memory formattedSubmissions = new string[](allSubmissions.length);
        for (uint i = 0; i < allSubmissions.length; i++) {
            formattedSubmissions[i] = bytes32ToString(allSubmissions[i]);
        }
        return formattedSubmissions;
    }

    function bytes32ToString(bytes32 _bytes32) private pure returns (string memory) {
        bytes memory bytesArray = new bytes(32);
        for (uint256 i; i < 32; i++) {
            bytesArray[i] = _bytes32[i];
        }
        return string(bytesArray);
    }

    // 新增任务申请函数
    function applyForTask(bytes memory encryptedData, uint256 requestedAmount) public {
        // 只触发事件而不存储数据
        emit TaskApplied(msg.sender, encryptedData, requestedAmount);
    }

    // 需求方查看所有任务申请并解密
    function viewTaskRequests() public view returns (bytes[] memory, uint256[] memory, address[] memory) {
        bytes[] memory encryptedDataArray = new bytes[](taskRequests.length);
        uint256[] memory requestedAmountArray = new uint256[](taskRequests.length);
        address[] memory applicantArray = new address[](taskRequests.length);

        for (uint i = 0; i < taskRequests.length; i++) {
            encryptedDataArray[i] = taskRequests[i].encryptedData;
            requestedAmountArray[i] = taskRequests[i].requestedAmount;
            applicantArray[i] = taskRequests[i].applicant;
        }

        return (encryptedDataArray, requestedAmountArray, applicantArray);
    }

    // RewardDistribution.sol 中的函数
    modifier onlyAfterDeadlineOrMaxSubmissions(uint256 task) {
        require(block.timestamp >= rewardCalculateDeadline || taskData[task].submissionCount >= maxRewardCalculateSubmissions, "Cannot call function before deadline or max submissions reached");
        _;
    }

    // 优化后的 submitResultHash 函数
    function submitResultHash(bytes32 dataHash) public {
        require(block.timestamp < rewardCalculateDeadline, "Current time is after deadline");
        require(taskData[0].submissionCount < maxRewardCalculateSubmissions, "Maximum submissions reached");

        Task storage t = taskData[0];

        // 如果提交者还没有提交过结果，则记录其提交，并初始化投票数为 1
        if (t.submitters[msg.sender] == bytes32(0)) {
            t.submitters[msg.sender] = dataHash;
            t.votes[dataHash] = 1;  // 初始投票为 1
            t.submissionCount++;

            // 检查并更新当前投票最多的哈希值
            if (t.votes[dataHash] > t.currentHighestVotes) {
                t.currentMostVotedHash = dataHash;
                t.currentHighestVotes = t.votes[dataHash];
            }

            emit TaskSubmitted(msg.sender, 0, dataHash);
        } else {
            // 如果该提交者已经提交过，则忽略后续提交
            revert("You have already submitted a result.");
        }
    }

    // 新增获取当前投票最多的哈希值的函数
    function getMostVotedDataHash() public view returns (bytes32) {
        return taskData[0].currentMostVotedHash;
    }

    // 新增存款功能，用户可以向合约存入保证金
    function deposit() public payable {
        require(msg.sender == owner, "Only the owner can deposit the funds");
        deposits[msg.sender] += msg.value;  // 记录用户的保证金余额
    }

    function rewardDistribution(
        address[] memory workers,
        uint256[] memory rewards
    ) public payable {
        require(workers.length == rewards.length, "Workers and rewards length mismatch");

        uint256 length = workers.length;
        bytes32 mostVotedHash = taskData[0].currentMostVotedHash;  // 获取当前投票最多的哈希值

        // 计算根据 workers 和 rewards 生成的哈希值
        bytes memory combinedData;
        for (uint i = 0; i < workers.length; i++) {
            combinedData = abi.encodePacked(combinedData, workers[i], rewards[i]);
        }
        bytes32 calculatedHash = keccak256(combinedData);

        // 验证生成的哈希值是否与当前最高投票哈希值一致
        require(calculatedHash == mostVotedHash, "Generated hash does not match the most voted hash");
        require(submittedWorkers == maxRewardCalculateSubmissions, "Not enough workers have submitted hashes");

        for (uint i = 0; i < length; i++) {
            (bool success, ) = workers[i].call{value: rewards[i]}("");
            require(success, "Reward transfer failed");
            emit WorkerReward(workers[i], rewards[i]);
        }
    }

}