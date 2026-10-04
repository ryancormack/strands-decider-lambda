import * as path from "node:path";
import * as cdk from "aws-cdk-lib";
import * as ecrAssets from "aws-cdk-lib/aws-ecr-assets";
import * as lambda from "aws-cdk-lib/aws-lambda";
import * as logs from "aws-cdk-lib/aws-logs";
import type { Construct } from "constructs";

export interface DeciderStackProps extends cdk.StackProps {
  /** Concurrent executions reserved for the function; each one holds ~8 GiB. */
  readonly reservedConcurrentExecutions?: number;
}

// 10,240 MB is Lambda's maximum and grants 6 vCPUs. The fp32 torso alone is ~7 GiB.
const MEMORY_MB = 10240;
const VCPUS = 6;

export class DeciderStack extends cdk.Stack {
  public readonly function: lambda.DockerImageFunction;
  public readonly functionUrl: lambda.FunctionUrl;

  constructor(scope: Construct, id: string, props: DeciderStackProps = {}) {
    super(scope, id, props);

    const logGroup = new logs.LogGroup(this, "LogGroup", {
      retention: logs.RetentionDays.ONE_MONTH,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
    });

    this.function = new lambda.DockerImageFunction(this, "Function", {
      code: lambda.DockerImageCode.fromImageAsset(path.join(__dirname, "..", "lambda"), {
        platform: ecrAssets.Platform.LINUX_ARM64,
        exclude: ["tests", "__pycache__", ".pytest_cache"],
      }),
      architecture: lambda.Architecture.ARM_64,
      memorySize: MEMORY_MB,
      // Covers a cold model load plus the request when init overruns Lambda's 10 s init cap.
      timeout: cdk.Duration.seconds(180),
      reservedConcurrentExecutions: props.reservedConcurrentExecutions ?? 5,
      logGroup,
      environment: {
        DECIDER_NUM_THREADS: String(VCPUS),
        OMP_NUM_THREADS: String(VCPUS),
      },
    });

    this.functionUrl = this.function.addFunctionUrl({
      authType: lambda.FunctionUrlAuthType.AWS_IAM,
    });

    new cdk.CfnOutput(this, "FunctionUrl", { value: this.functionUrl.url });
    new cdk.CfnOutput(this, "FunctionName", { value: this.function.functionName });
  }
}
