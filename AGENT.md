# RGB project directory guidance

Both the remote machine and the user's local machine have their own RGB project
directory. They are separate checkouts, not a shared filesystem.

- Current remote checkout: `/home/ref/RGB_SFM_V3` on remote140165.
- Local Windows/WSL checkout: exists; its exact absolute path is not yet verified.
- Local artifact inbox: `E:\RGB_SFM_artifacts` (`/mnt/e/RGB_SFM_artifacts` in WSL).

Verify the machine and project root before transferring or extracting files.
An archive in the inbox is not yet restored into the local project's `plots/`.
Never report a remote-container operation as a change to the local checkout.

See [AGENTS.md](AGENTS.md) for the full project instructions and transfer protocol.
