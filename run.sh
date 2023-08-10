# 设置环境变量

export WECHATY_LOG="info"
export WECHATY_PUPPET="wechaty-puppet-padlocal"
export WECHATY_PUPPET_PADLOCAL_TOKEN="puppet_padlocal_56a4556ac797489783a6d1f2d50ccb4c"

export WECHATY_PUPPET_SERVER_PORT="9001"
# 可使用代码随机生成UUID：import uuid;print(uuid.uuid4());
export WECHATY_TOKEN="1fe5f846-3cfb-401d-b20c-XXXXX"

docker run -ti \
  --name wechaty_puppet_service_token_gateway \
  --rm \
  -e WECHATY_LOG \
  -e WECHATY_PUPPET \
  -e WECHATY_PUPPET_PADLOCAL_TOKEN \
  -e WECHATY_PUPPET_SERVER_PORT \
  -e WECHATY_TOKEN \
  -p "$WECHATY_PUPPET_SERVER_PORT:$WECHATY_PUPPET_SERVER_PORT" \
  wechaty/wechaty:0.56
