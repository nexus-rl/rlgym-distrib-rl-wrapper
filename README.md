# RLGym Wrapper for distrib-rl

This module is a simple wrapper for the [rlgym](https://rlgym.org) environment
that makes it compatible with the
[distrib-rl](https://github.com/AechPro/distrib-rl) project.

## Project goals

This library exists to handle two special concerns:

1. Compatibility with the distrib-rl config system
2. Compatibility with the latest gym environment specification (v0.25.x at time
   of writing)

## API stability

For the moment this library should be considered highly unstable, as our first
official release is not yet planned. As the various components of `distrib-rl`
stabilize, so too with this library, however as long as this library is
maintained it will continue to be updated to track the latest versions of `gym`
(or whatever reinforcement learning environment specification `distrib-rl`
requires) and `distrib-rl`

### Versioning

Once published, this library will make _strict_ use of the [semver 2.0.0
versioning specification](https://semver.org/spec/v2.0.0.html), and those
depending on this library in their own projects should take care to set their
version ranges appropriately.

#### Breaking changes and Public API

Due to the goals stated above, this project is at the mercy of the gym API with
respect to its stability. That is, we cannot meet the goal of tracking the
latest gym environment version if we limit our ability to make breaking changes
to this environment. As a result, expect that breaking changes may be made to
this library at any time. However as stated above, we will always communicate
the presence of these changes in the way prescribed by the semver versioning
standard.


For the moment the only portions of our API that we consider to be public are
the (currently undocumented) definitions of the dict structures for distrib-rl
environment config, and the portions of this API that are extensions of the Gym
`Env` class.

As we move closer to an official public release, we will endeavor to document
the config API via JSONSchema or similar.

