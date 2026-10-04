#!/usr/bin/env node
import * as cdk from "aws-cdk-lib";
import { DeciderStack } from "../lib/decider-stack.js";

const app = new cdk.App();

new DeciderStack(app, "StrandsDeciderLambda", {
  env: {
    account: process.env.CDK_DEFAULT_ACCOUNT,
    region: process.env.CDK_DEFAULT_REGION ?? "eu-west-1",
  },
});
