from crewai import Agent, Task, Crew, LLM
from crewai_tools.aws.bedrock.agents.invoke_agent_tool import BedrockInvokeAgentTool
import boto3
import os

# Disable CrewAI telemetry
os.environ["CREWAI_TRACKING"] = "False"

def get_aws_credentials():
    """Get AWS credentials from the environment"""
    session = boto3.Session()
    credentials = session.get_credentials()
    return {
        "aws_access_key_id": credentials.access_key,
        "aws_secret_access_key": credentials.secret_key,
        "aws_session_token": credentials.token
    }

def create_bedrock_agent():
    """Create and return the Bedrock agent"""
    # Get AWS credentials
    credentials = get_aws_credentials()
    
    # Initialize the LLM
    llm = LLM(
        model="bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0",
        aws_access_key_id=credentials["aws_access_key_id"],
        aws_secret_access_key=credentials["aws_secret_access_key"],
        aws_session_token=credentials["aws_session_token"],
        aws_region_name='us-west-2'
    )
    
    # Create the Bedrock agent tool
    agent_tool = BedrockInvokeAgentTool(
        agent_id="ANBKZJFQKQ",
        agent_alias_id="UAZCDSO1NK"
    )
    
    # Create an agent that uses the Bedrock agent tool
    aws_expert = Agent(
        role='AI Agent Expert',
        goal='Help users understand AI Agentic Memory',
        backstory='I am an expert in Agentic AI and Agentic Memory',
        tools=[agent_tool],
        llm=llm,
        verbose=False  # Changed from True to False to reduce telemetry
    )
    
    return aws_expert

def run_agent_with_query(query):
    """Run the agent with a specific query and return the result"""
    # Create the agent
    aws_expert = create_bedrock_agent()
    
    # Create a task for the agent
    task = Task(
        description=f"explain {query}",
        expected_output="A comprehensive explanation",
        agent=aws_expert
    )
    
    # Create a crew with the agent
    crew = Crew(
        agents=[aws_expert],
        tasks=[task],
        verbose=False  # Changed from True to False to reduce telemetry
    )
    
    # Run the crew
    result = crew.kickoff()
    
    return result

# For testing the module directly
if __name__ == "__main__":
    test_query = "What is agentic memory?"
    print(f"Testing with query: {test_query}")
    result = run_agent_with_query(test_query)
    print(f"Result: {result}")
