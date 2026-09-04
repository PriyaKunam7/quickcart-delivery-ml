# Database Restart Runbook

## Overview
This runbook covers safe restart procedures for the primary PostgreSQL
database cluster used by the CMDB service.

## Preconditions
Confirm there is no active incident before restarting. Check the
observability dashboard for current replication lag.

### Checking replication lag
Run the replication status query against the primary node and confirm
lag is under 5 seconds before proceeding.

## Restart Steps
1. Notify the on-call channel.
2. Drain traffic from the node.
3. Perform the restart.
4. Verify health checks pass.

## Rollback
If health checks fail after restart, fail back to the standby replica
immediately and escalate to the database team.