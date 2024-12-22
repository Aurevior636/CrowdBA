// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/proxy/Clones.sol";
import "./TaskContract.sol";

contract Factory {
    using Clones for address;

    event TaskContractCreated(address indexed creator, address taskContract, uint256 taskId);

    uint256 public taskCount;
    mapping(uint256 => address) public taskContracts;
    mapping(address => uint256[]) public creatorTasks;
    address public owner;

    address public immutable taskContractImplementation;

    constructor(address _taskContractImplementation) {
        owner = msg.sender;
        taskContractImplementation = _taskContractImplementation;
    }

    modifier onlyOwner() {
        require(msg.sender == owner, "Only the contract owner can call this function");
        _;
    }

    // 修改createTaskContract函数，添加depositAmount参数
    function createTaskContract(
        uint256 _SubmitDataDURATION,
        uint256 _maxSubmissions,
        uint256 _rewardDuration,
        uint256 _rewardMaxSubmissions,
        string memory _ipfsHash,
        uint256 _depositAmount  // 新增参数，表示保证金数量
    ) public {
        // 克隆任务合约
        address clone = taskContractImplementation.clone();
        TaskContract newTaskContract = TaskContract(clone);

        // 初始化任务合约，并传入所有必要的参数
        newTaskContract.initialize(
            msg.sender,
            _SubmitDataDURATION,
            _maxSubmissions,
            _rewardDuration,
            _rewardMaxSubmissions,
            _ipfsHash,
            _depositAmount  // 传递保证金数量
        );

        // 存储任务合约地址
        taskContracts[taskCount] = address(newTaskContract);
        // 记录创建者的任务
        creatorTasks[msg.sender].push(taskCount);

        // 发出任务合约创建事件
        emit TaskContractCreated(msg.sender, address(newTaskContract), taskCount);
        
        // 增加任务计数
        taskCount++;
    }

    function getTaskContractAddress(uint256 taskId) public view returns (address) {
        return taskContracts[taskId];
    }

    function getTasksByCreator(address creator) public view returns (uint256[] memory) {
        return creatorTasks[creator];
    }
}
