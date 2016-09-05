----------------------------------------------------------------------------------
---- Basic web service using turbo, to caption images using a pre-trained network
----
------------------------------------------------------------------------------------

local turbo = require("turbo")
local torch = require("torch")
local image = require("image")
local nn = require("nn")
local socket = require("socket")

