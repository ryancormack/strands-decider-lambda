import * as cdk from "aws-cdk-lib";
import { Match, Template } from "aws-cdk-lib/assertions";
import { describe, expect, it } from "vitest";
import { DeciderStack } from "../lib/decider-stack.js";

function template(props = {}): Template {
  const app = new cdk.App();
  const stack = new DeciderStack(app, "Test", { env: { account: "111111111111", region: "eu-west-1" }, ...props });
  return Template.fromStack(stack);
}

describe("DeciderStack", () => {
  const t = template();

  it("runs a container image on arm64 at the Lambda maximum memory", () => {
    t.hasResourceProperties("AWS::Lambda::Function", {
      PackageType: "Image",
      Architectures: ["arm64"],
      MemorySize: 10240,
      Timeout: 180,
      ReservedConcurrentExecutions: 5,
      Environment: { Variables: { DECIDER_NUM_THREADS: "6", OMP_NUM_THREADS: "6" } },
    });
  });

  it("exposes exactly one Function URL and it uses IAM auth", () => {
    t.resourceCountIs("AWS::Lambda::Url", 1);
    t.hasResourceProperties("AWS::Lambda::Url", { AuthType: "AWS_IAM" });
  });

  it("grants no public invoke permission", () => {
    const permissions = t.findResources("AWS::Lambda::Permission");
    for (const permission of Object.values(permissions)) {
      expect(permission.Properties.Principal).not.toBe("*");
      expect(permission.Properties.FunctionUrlAuthType).not.toBe("NONE");
    }
  });

  it("ships logs to a retained-for-a-month log group that is destroyed with the stack", () => {
    t.hasResource("AWS::Logs::LogGroup", {
      Properties: { RetentionInDays: 30 },
      DeletionPolicy: "Delete",
    });
  });

  it("exports the URL", () => {
    t.hasOutput("FunctionUrl", { Value: Match.anyValue() });
  });

  it("lets the reserved concurrency be overridden", () => {
    template({ reservedConcurrentExecutions: 2 }).hasResourceProperties("AWS::Lambda::Function", {
      ReservedConcurrentExecutions: 2,
    });
  });
});
